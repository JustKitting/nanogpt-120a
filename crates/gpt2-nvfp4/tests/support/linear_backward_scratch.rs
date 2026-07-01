#![allow(dead_code)]

use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{AttentionCProjScratch, AttentionQkvScratch, GPT2_TOKEN_ROWS};
use rust_kernels_cuda::linear_backward::LinearBackwardMsEdenScratchBuffers as MsEdenScratch;

pub struct LinearBackwardScratchBuffers {
    error_t: DeviceBuffer<f32>,
    weight_t: DeviceBuffer<f32>,
    input_t: DeviceBuffer<f32>,
    linear: MsEdenScratch,
}

impl LinearBackwardScratchBuffers {
    pub fn new(stream: &CudaStream, input: usize, output: usize) -> Result<Self, DriverError> {
        Ok(Self {
            error_t: DeviceBuffer::zeroed(stream, output * GPT2_TOKEN_ROWS)?,
            weight_t: DeviceBuffer::zeroed(stream, output * input)?,
            input_t: DeviceBuffer::zeroed(stream, input * GPT2_TOKEN_ROWS)?,
            linear: MsEdenScratch::new(stream, GPT2_TOKEN_ROWS, input, output)?,
        })
    }

    pub fn c_proj(&mut self) -> AttentionCProjScratch<'_> {
        AttentionCProjScratch {
            error_t: &mut self.error_t,
            weight_t: &mut self.weight_t,
            input_t: &mut self.input_t,
            linear: self.linear.as_args(),
        }
    }

    pub fn qkv(&mut self) -> AttentionQkvScratch<'_> {
        AttentionQkvScratch {
            error_t: &mut self.error_t,
            weight_t: &mut self.weight_t,
            input_t: &mut self.input_t,
            linear: self.linear.as_args(),
        }
    }
}
