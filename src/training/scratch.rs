use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    AttentionCoreScratch, BlockAttentionBackwardScratch, GPT2_MLP, GPT2_N_EMBD, GPT2_QKV,
    GPT2_VOCAB_SIZE, Gpt2BackwardScratch, MlpBackwardScratch,
};

use super::linear_scratch::LinearScratch;

pub struct BackwardScratchBuffers {
    final_head: LinearScratch,
    attention_c_proj: LinearScratch,
    attention_qkv: LinearScratch,
    attention_softmax_d: DeviceBuffer<f32>,
    mlp_down: LinearScratch,
    mlp_up: LinearScratch,
}

impl BackwardScratchBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            final_head: LinearScratch::new(stream, GPT2_N_EMBD, GPT2_VOCAB_SIZE)?,
            attention_c_proj: LinearScratch::new(stream, GPT2_N_EMBD, GPT2_N_EMBD)?,
            attention_qkv: LinearScratch::new(stream, GPT2_N_EMBD, GPT2_QKV)?,
            attention_softmax_d: DeviceBuffer::zeroed(stream, gpt2_nvfp4::AttentionLogSumExp::LEN)?,
            mlp_down: LinearScratch::new(stream, GPT2_MLP, GPT2_N_EMBD)?,
            mlp_up: LinearScratch::new(stream, GPT2_N_EMBD, GPT2_MLP)?,
        })
    }

    pub fn scratch(&mut self) -> Gpt2BackwardScratch<'_> {
        let (down_error_t, down_weight_t, down_input_t, down_linear) = self.mlp_down.parts();
        let (up_error_t, up_weight_t, up_input_t, up_linear) = self.mlp_up.parts();
        Gpt2BackwardScratch {
            final_head: self.final_head.final_head(),
            attention: BlockAttentionBackwardScratch {
                c_proj: self.attention_c_proj.attention(),
                core: AttentionCoreScratch {
                    softmax_d: &mut self.attention_softmax_d,
                },
                qkv: self.attention_qkv.attention(),
            },
            mlp: MlpBackwardScratch {
                down_error_t,
                down_weight_t,
                down_input_t,
                up_error_t,
                up_weight_t,
                up_input_t,
                down_linear,
                up_linear,
            },
        }
    }
}
