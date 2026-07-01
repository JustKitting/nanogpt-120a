use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_TOKEN_ROWS, HiddenState, LayerNormSaved, LayerNormTape};

use super::device_buffer::zero;
pub(super) use gpt2_nvfp4::RowwiseNvfp4Buffers as RowwiseTapeBuffers;

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
