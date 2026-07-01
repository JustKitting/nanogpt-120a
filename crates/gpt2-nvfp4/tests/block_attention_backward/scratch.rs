use cuda_core::{CudaStream, DriverError};
use gpt2_nvfp4::{
    AttentionCoreScratchBuffers, BlockAttentionBackwardScratch, GPT2_N_EMBD, GPT2_QKV,
    LinearScratch,
};

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
            c_proj: self.c_proj.c_proj(),
            core: self.core.args(),
            qkv: self.qkv.qkv(),
        }
    }
}
