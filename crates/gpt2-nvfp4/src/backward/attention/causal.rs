use cuda_core::DriverError;
use rust_kernels_cuda::attention::CausalAttentionBackwardArgs;

use super::types::AttentionCoreBackwardArgs;
use crate::{GPT2_CONTEXT_LEN, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_QKV};

pub fn causal_attention_backward(
    args: AttentionCoreBackwardArgs<'_, '_, '_>,
) -> Result<(), DriverError> {
    args.module
        .causal_attention_backward(CausalAttentionBackwardArgs {
            stream: args.stream,
            qkv: args.saved.qkv,
            attention_out: args.saved.attention_out,
            d_out: args.d_attention_out,
            lse: args.saved.attention_lse,
            softmax_d: args.scratch.softmax_d,
            d_qkv: args.d_qkv,
            token_count: GPT2_CONTEXT_LEN as u32,
            embedding_dim: GPT2_N_EMBD as u32,
            qkv_dim: GPT2_QKV as u32,
            head_count: GPT2_N_HEAD as u32,
            head_dim: (GPT2_N_EMBD / GPT2_N_HEAD) as u32,
        })
}
