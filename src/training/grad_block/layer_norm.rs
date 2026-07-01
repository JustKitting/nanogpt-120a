use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{HiddenState, LayerNormGrads, GPT2_N_EMBD};

use crate::training::device_buffer::zero;

pub struct LayerNormGradBuffers {
    pub(in crate::training) d_residual: DeviceBuffer<f32>,
    pub(in crate::training) d_normalized: DeviceBuffer<f32>,
    pub(in crate::training) d_weight: DeviceBuffer<f32>,
    pub(in crate::training) d_bias: DeviceBuffer<f32>,
}

impl LayerNormGradBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            d_residual: zero(stream, HiddenState::LEN)?,
            d_normalized: zero(stream, HiddenState::LEN)?,
            d_weight: zero(stream, GPT2_N_EMBD)?,
            d_bias: zero(stream, GPT2_N_EMBD)?,
        })
    }

    pub fn grads(&mut self) -> LayerNormGrads<'_> {
        LayerNormGrads {
            d_residual: &mut self.d_residual,
            d_normalized: &mut self.d_normalized,
            d_weight: &mut self.d_weight,
            d_bias: &mut self.d_bias,
        }
    }
}
