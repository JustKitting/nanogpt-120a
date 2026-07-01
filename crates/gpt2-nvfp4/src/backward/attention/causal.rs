use cuda_core::DriverError;
use rust_kernels_cuda::attention::CausalAttentionBackwardTcArgs;

use super::types::AttentionCoreBackwardArgs;
use crate::{Gpt2Config, GPT2_N_EMBD, GPT2_N_HEAD};

pub fn causal_attention_backward(
    args: AttentionCoreBackwardArgs<'_, '_, '_>,
) -> Result<(), DriverError> {
    let qkv_dim = Gpt2Config::attention_qkv_dim(args.use_full_attention) as u32;
    let tc_args = CausalAttentionBackwardTcArgs {
        stream: args.stream,
        tc_module: args.tc_module,
        qkv: args.saved.qkv,
        attention_out: args.saved.attention_out,
        d_out: args.d_attention_out,
        log_sum_exp: args.saved.attention_log_sum_exp,
        softmax_d: args.scratch.softmax_d,
        d_qkv: args.d_qkv,
        scratch: args.scratch.tc,
        row_count: args.saved.row_count,
        seq_len: args.saved.seq_len,
        batch_size: args.saved.batch_size,
        embedding_dim: GPT2_N_EMBD as u32,
        qkv_dim,
        head_count: GPT2_N_HEAD as u32,
        head_dim: (GPT2_N_EMBD / GPT2_N_HEAD) as u32,
    };
    if args.use_full_attention {
        args.module.causal_attention_backward_tc(tc_args)
    } else {
        args.module.kda_attention_backward_tc(tc_args)
    }
}
