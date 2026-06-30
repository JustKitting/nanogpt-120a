use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_TOKEN_ROWS, HiddenState, LayerNormSaved, LayerNormTape, RowwiseNvfp4Tape};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use super::device_buffer::zero;

pub struct LayerNormTapeBuffers {
    residual: DeviceBuffer<u16>,
    mean: DeviceBuffer<f32>,
    inv_std: DeviceBuffer<f32>,
}

impl LayerNormTapeBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            residual: zero(stream, HiddenState::LEN)?,
            mean: zero(stream, GPT2_TOKEN_ROWS)?,
            inv_std: zero(stream, GPT2_TOKEN_ROWS)?,
        })
    }

    pub fn tape(&mut self) -> LayerNormTape<'_> {
        LayerNormTape {
            residual: &mut self.residual,
            mean: &mut self.mean,
            inv_std: &mut self.inv_std,
        }
    }

    pub fn saved(&self, row_count: u32) -> LayerNormSaved<'_> {
        LayerNormSaved {
            row_count,
            residual: &self.residual,
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
            bytes: zero(stream, elements / 2)?,
            scales: zero(stream, elements / 16)?,
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
