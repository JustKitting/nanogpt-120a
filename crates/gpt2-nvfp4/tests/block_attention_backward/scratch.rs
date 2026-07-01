use cuda_core::{CudaStream, DriverError};
use gpt2_nvfp4::{
    AttentionCoreScratchBuffers, BlockAttentionBackwardScratch, GPT2_N_EMBD, GPT2_QKV,
};

use crate::common::linear_backward_scratch::LinearBackwardScratchBuffers;

pub struct BlockAttentionScratch {
    c_proj: LinearBackwardScratchBuffers,
    qkv: LinearBackwardScratchBuffers,
    core: AttentionCoreScratchBuffers,
}

impl BlockAttentionScratch {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            c_proj: LinearBackwardScratchBuffers::new(stream, GPT2_N_EMBD, GPT2_N_EMBD)?,
            qkv: LinearBackwardScratchBuffers::new(stream, GPT2_N_EMBD, GPT2_QKV)?,
            core: AttentionCoreScratchBuffers::new(stream)?,
        })
    }

    pub fn block(&mut self) -> BlockAttentionBackwardScratch<'_> {
        BlockAttentionBackwardScratch {
            c_proj: self.c_proj.c_proj(),
            core: self.core.args(),
            qkv: self.qkv.qkv(),
        }
    }
}
