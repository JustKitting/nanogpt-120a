use cuda_core::{DriverError, LaunchConfig};

use super::gather::TC_FORWARD_THREADS_PER_BLOCK;
use super::types::CausalAttentionTcArgs;
use crate::attention::{AttentionModule, CausalAttentionParams};
use crate::f16_tc_matmul::{F16TcMatmulF32Args, F16TcMatmulF32RhsArgs};

impl AttentionModule {
    pub fn causal_attention_tc(
        &self,
        args: CausalAttentionTcArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        let params = CausalAttentionParams {
            row_count: args.row_count,
            seq_len: args.seq_len,
            batch_size: args.batch_size,
            embedding_dim: args.embedding_dim,
            qkv_dim: args.qkv_dim,
            head_count: args.head_count,
            head_dim: args.head_dim,
            scale: 1.0 / (args.head_dim as f32).sqrt(),
        };
        let batch_head = args.batch_size * args.head_count;
        let scratch = args.scratch;

        self.causal_attention_tc.gather_qkv_forward_kernel(
            args.stream,
            linear_config(batch_head * args.seq_len * args.head_dim),
            args.qkv,
            &mut *scratch.q,
            &mut *scratch.k,
            &mut *scratch.v,
            params,
        )?;
        args.tc_module
            .batched_matmul_f32_input(F16TcMatmulF32Args {
                stream: args.stream,
                a: &*scratch.q,
                b_t: &*scratch.k,
                out: &mut *scratch.scores,
                batch_count: batch_head,
                m: args.seq_len,
                n: args.seq_len,
                k: args.head_dim,
            })?;
        self.causal_attention_tc.attention_softmax_forward_kernel(
            args.stream,
            attention_config(args.seq_len, args.head_count, args.batch_size),
            &*scratch.scores,
            &mut *scratch.probs,
            args.log_sum_exp,
            params,
        )?;
        args.tc_module
            .batched_matmul_f32_rhs(F16TcMatmulF32RhsArgs {
                stream: args.stream,
                a: &*scratch.probs,
                rhs: &*scratch.v,
                out: &mut *scratch.compact_out,
                batch_count: batch_head,
                m: args.seq_len,
                n: args.head_dim,
                k: args.seq_len,
            })?;
        self.causal_attention_tc.scatter_attention_forward_kernel(
            args.stream,
            linear_config(batch_head * args.seq_len * args.head_dim),
            &*scratch.compact_out,
            args.out,
            params,
        )
    }
}

fn linear_config(element_count: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (element_count.div_ceil(TC_FORWARD_THREADS_PER_BLOCK), 1, 1),
        block_dim: (TC_FORWARD_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn attention_config(seq_len: u32, head_count: u32, batch_size: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (seq_len, head_count, batch_size),
        block_dim: (TC_FORWARD_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}
