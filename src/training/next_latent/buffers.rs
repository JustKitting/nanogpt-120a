use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_TOKEN_ROWS, HiddenState, NextLatHiddenActivation, NextLatInputActivation};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use super::super::device_buffer::zero;

pub struct NextLatBuffers {
    pub(super) next_token_embeddings: DeviceBuffer<f32>,
    pub(super) concat: DeviceBuffer<f32>,
    pub(super) normalized: DeviceBuffer<f32>,
    pub(super) normalized_amax: DeviceBuffer<f32>,
    pub(super) mean: DeviceBuffer<f32>,
    pub(super) inv_std: DeviceBuffer<f32>,
    pub(super) input_quant: NextLatRowwiseBuffers,
    pub(super) pre1: DeviceBuffer<f32>,
    pub(super) act1: DeviceBuffer<f32>,
    pub(super) act1_quant: NextLatRowwiseBuffers,
    pub(super) pre2: DeviceBuffer<f32>,
    pub(super) act2: DeviceBuffer<f32>,
    pub(super) act2_quant: NextLatRowwiseBuffers,
    pub(super) delta: DeviceBuffer<f32>,
    pub(super) predicted: DeviceBuffer<f32>,
    pub(super) losses: DeviceBuffer<f32>,
    pub(super) d_predicted: DeviceBuffer<f32>,
}

pub(super) struct NextLatRowwiseBuffers {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scales: DeviceBuffer<f32>,
}

pub(super) struct RowwiseOut<'a> {
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scales: &'a mut DeviceBuffer<f32>,
}

pub(super) struct RowwiseQuantizeBuffers<'a> {
    pub input: &'a DeviceBuffer<f32>,
    pub amax: &'a mut DeviceBuffer<f32>,
    pub out: RowwiseOut<'a>,
}

impl NextLatBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            next_token_embeddings: zero(stream, HiddenState::LEN)?,
            concat: zero(stream, NextLatInputActivation::LEN)?,
            normalized: zero(stream, NextLatInputActivation::LEN)?,
            normalized_amax: zero(stream, GPT2_TOKEN_ROWS)?,
            mean: zero(stream, GPT2_TOKEN_ROWS)?,
            inv_std: zero(stream, GPT2_TOKEN_ROWS)?,
            input_quant: NextLatRowwiseBuffers::new(stream, NextLatInputActivation::LEN)?,
            pre1: zero(stream, NextLatHiddenActivation::LEN)?,
            act1: zero(stream, NextLatHiddenActivation::LEN)?,
            act1_quant: NextLatRowwiseBuffers::new(stream, NextLatHiddenActivation::LEN)?,
            pre2: zero(stream, NextLatHiddenActivation::LEN)?,
            act2: zero(stream, NextLatHiddenActivation::LEN)?,
            act2_quant: NextLatRowwiseBuffers::new(stream, NextLatHiddenActivation::LEN)?,
            delta: zero(stream, HiddenState::LEN)?,
            predicted: zero(stream, HiddenState::LEN)?,
            losses: zero(stream, GPT2_TOKEN_ROWS)?,
            d_predicted: zero(stream, HiddenState::LEN)?,
        })
    }

    pub(crate) fn losses(&self) -> &DeviceBuffer<f32> {
        &self.losses
    }

    pub(super) fn input_quantize(&mut self) -> RowwiseQuantizeBuffers<'_> {
        RowwiseQuantizeBuffers {
            input: &self.normalized,
            amax: &mut self.normalized_amax,
            out: self.input_quant.out(),
        }
    }

    pub(super) fn act1_quantize(&mut self) -> RowwiseQuantizeBuffers<'_> {
        RowwiseQuantizeBuffers {
            input: &self.act1,
            amax: &mut self.normalized_amax,
            out: self.act1_quant.out(),
        }
    }

    pub(super) fn act2_quantize(&mut self) -> RowwiseQuantizeBuffers<'_> {
        RowwiseQuantizeBuffers {
            input: &self.act2,
            amax: &mut self.normalized_amax,
            out: self.act2_quant.out(),
        }
    }
}

impl NextLatRowwiseBuffers {
    fn new(stream: &CudaStream, len: usize) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: zero(stream, len / 2)?,
            scales: zero(stream, len / 16)?,
            global_scales: zero(stream, GPT2_TOKEN_ROWS)?,
        })
    }

    pub(super) fn rowwise(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor::new(&self.bytes, &self.scales, &self.global_scales)
    }

    fn out(&mut self) -> RowwiseOut<'_> {
        RowwiseOut {
            bytes: &mut self.bytes,
            scales: &mut self.scales,
            global_scales: &mut self.global_scales,
        }
    }
}
