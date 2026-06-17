use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_CONTEXT_LEN, HiddenState, LayerNormSaved, LayerNormTape, RowwiseNvfp4Tape};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

pub struct LayerNormTapeBuffers {
    residual: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    mean: DeviceBuffer<f32>,
    inv_std: DeviceBuffer<f32>,
}

impl LayerNormTapeBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            residual: zero(stream, HiddenState::LEN)?,
            normalized: zero(stream, HiddenState::LEN)?,
            mean: zero(stream, GPT2_CONTEXT_LEN)?,
            inv_std: zero(stream, GPT2_CONTEXT_LEN)?,
        })
    }

    pub fn tape(&mut self) -> LayerNormTape<'_> {
        LayerNormTape {
            residual: &mut self.residual,
            normalized: &mut self.normalized,
            mean: &mut self.mean,
            inv_std: &mut self.inv_std,
        }
    }

    pub fn saved(&self) -> LayerNormSaved<'_> {
        LayerNormSaved {
            residual: &self.residual,
            normalized: &self.normalized,
            mean: &self.mean,
            inv_std: &self.inv_std,
        }
    }
}

pub struct RowwiseTapeBuffers {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scales: DeviceBuffer<f32>,
}

impl RowwiseTapeBuffers {
    pub fn new(stream: &CudaStream, elements: usize, rows: usize) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, elements / 2)?,
            scales: DeviceBuffer::zeroed(stream, elements / 16)?,
            global_scales: zero(stream, rows)?,
        })
    }

    pub fn tape(&mut self) -> RowwiseNvfp4Tape<'_> {
        RowwiseNvfp4Tape {
            bytes: &mut self.bytes,
            scales: &mut self.scales,
            global_scales: &mut self.global_scales,
        }
    }

    pub fn saved(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor {
            bytes: &self.bytes,
            scales: &self.scales,
            global_scales: &self.global_scales,
        }
    }
}

pub fn zero(stream: &CudaStream, len: usize) -> Result<DeviceBuffer<f32>, DriverError> {
    DeviceBuffer::zeroed(stream, len)
}
