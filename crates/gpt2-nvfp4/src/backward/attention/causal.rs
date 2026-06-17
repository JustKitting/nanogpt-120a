use cuda_core::DriverError;
use rust_kernels_cuda::attention::CausalAttentionBackwardArgs;

use super::types::AttentionCoreBackwardArgs;
use crate::{GPT2_N_EMBD, GPT2_N_HEAD, GPT2_QKV};

pub fn causal_attention_backward(
    args: AttentionCoreBackwardArgs<'_, '_, '_>,
) -> Result<(), DriverError> {
    args.module
        .causal_attention_backward(CausalAttentionBackwardArgs {
            stream: args.stream,
            qkv: args.saved.qkv,
            attention_out: args.saved.attention_out,
            d_out: args.d_attention_out,
            log_sum_exp: args.saved.attention_log_sum_exp,
            softmax_d: args.scratch.softmax_d,
            d_qkv: args.d_qkv,
            row_count: args.saved.row_count,
            seq_len: args.saved.seq_len,
            batch_size: args.saved.batch_size,
            embedding_dim: GPT2_N_EMBD as u32,
            qkv_dim: GPT2_QKV as u32,
            head_count: GPT2_N_HEAD as u32,
            head_dim: (GPT2_N_EMBD / GPT2_N_HEAD) as u32,
        })
}
