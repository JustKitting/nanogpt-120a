use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    AttentionCProjScratch, BlockAttentionBackwardScratch, GPT2_N_EMBD, GPT2_QKV, GPT2_TOKEN_ROWS,
};
use rust_kernels_cuda::linear_backward::LinearBackwardMsEdenScratchBuffers;

use super::attention_core_scratch::AttentionCoreScratchBuffers;

pub struct BlockAttentionScratch {
    c_proj: LinearScratch,
    qkv: LinearScratch,
    core: AttentionCoreScratchBuffers,
}

impl BlockAttentionScratch {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            c_proj: LinearScratch::new(stream, GPT2_N_EMBD, GPT2_N_EMBD)?,
            qkv: LinearScratch::new(stream, GPT2_N_EMBD, GPT2_QKV)?,
            core: AttentionCoreScratchBuffers::new(stream)?,
        })
    }

    pub fn block(&mut self) -> BlockAttentionBackwardScratch<'_> {
        BlockAttentionBackwardScratch {
            c_proj: self.c_proj.as_attention_scratch(),
            core: self.core.args(),
            qkv: self.qkv.as_attention_scratch(),
        }
    }
}

struct LinearScratch {
    error_t: DeviceBuffer<f32>,
    weight_t: DeviceBuffer<f32>,
    input_t: DeviceBuffer<f32>,
    linear: LinearBackwardMsEdenScratchBuffers,
}

impl LinearScratch {
    fn new(stream: &CudaStream, input_dim: usize, output_dim: usize) -> Result<Self, DriverError> {
        Ok(Self {
            error_t: DeviceBuffer::zeroed(stream, output_dim * GPT2_TOKEN_ROWS)?,
            weight_t: DeviceBuffer::zeroed(stream, output_dim * input_dim)?,
            input_t: DeviceBuffer::zeroed(stream, input_dim * GPT2_TOKEN_ROWS)?,
            linear: LinearBackwardMsEdenScratchBuffers::new(
                stream,
                GPT2_TOKEN_ROWS,
                input_dim,
                output_dim,
            )?,
        })
    }

    fn as_attention_scratch(&mut self) -> AttentionCProjScratch<'_> {
        AttentionCProjScratch {
            error_t: &mut self.error_t,
            weight_t: &mut self.weight_t,
            input_t: &mut self.input_t,
            linear: self.linear.as_args(),
        }
    }
}
