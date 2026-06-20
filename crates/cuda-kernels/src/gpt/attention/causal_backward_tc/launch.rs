use cuda_core::DriverError;

use super::launch_config::{attention_config, linear_config, tc_params};
use super::launch_grads::run_grad_matmuls;
use super::launch_scores::run_pair_scores;
use super::launch_transpose::{TransposeShape, run_transposes};
use super::matmul::AttentionTcMatmulContext;
use super::types::CausalAttentionBackwardTcArgs;
use crate::attention::AttentionModule;

impl AttentionModule {
    pub fn causal_attention_backward_tc(
        &self,
        args: CausalAttentionBackwardTcArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        let CausalAttentionBackwardTcArgs {
            stream,
            tc_module,
            qkv,
            attention_out,
            d_out,
            log_sum_exp,
            softmax_d,
            d_qkv,
            scratch,
            row_count,
            seq_len,
            batch_size,
            embedding_dim,
            qkv_dim,
            head_count,
            head_dim,
        } = args;
        let params = tc_params(
            row_count,
            seq_len,
            batch_size,
            embedding_dim,
            qkv_dim,
            head_count,
            head_dim,
        );
        let batch_head = batch_size * head_count;
        let tc_ctx = AttentionTcMatmulContext {
            stream,
            tc_module,
            batch_head,
            seq_len,
            head_dim,
        };
        let mut scratch = scratch;

        self.causal_attention_backward_tc.softmax_d_kernel(
            stream,
            attention_config(seq_len, head_count, batch_size),
            attention_out,
            d_out,
            softmax_d,
            params,
        )?;
        self.causal_attention_backward_tc.gather_qkv_dout_kernel(
            stream,
            linear_config(batch_head * seq_len * head_dim),
            qkv,
            d_out,
            scratch.q,
            scratch.k,
            scratch.v,
            scratch.d_out,
            params,
        )?;
        run_pair_scores(&tc_ctx, &mut scratch)?;
        self.causal_attention_backward_tc.attention_prob_ds_kernel(
            stream,
            linear_config(batch_head * seq_len * seq_len),
            scratch.scores,
            scratch.dot,
            log_sum_exp,
            softmax_d,
            scratch.p,
            scratch.ds,
            params,
        )?;
        run_transposes(
            &self.causal_attention_backward_tc,
            stream,
            &mut scratch,
            TransposeShape {
                batch_head,
                seq_len,
            },
        )?;
        run_grad_matmuls(&tc_ctx, &mut scratch)?;
        self.causal_attention_backward_tc.scatter_dqkv_kernel(
            stream,
            linear_config(batch_head * seq_len * head_dim),
            scratch.d_q,
            scratch.d_k,
            scratch.d_v,
            d_qkv,
            params,
        )
    }
}
