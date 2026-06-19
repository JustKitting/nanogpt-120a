use cuda_core::DriverError;

use super::matmul::{AttentionTcMatmulContext, run_tc_matmul};
use super::types::CausalAttentionBackwardTcScratch;

pub(super) fn run_grad_matmuls(
    ctx: &AttentionTcMatmulContext<'_>,
    scratch: &mut CausalAttentionBackwardTcScratch<'_>,
) -> Result<(), DriverError> {
    run_tc_matmul(
        ctx.stream,
        ctx.tc_module,
        scratch.ds,
        scratch.k_t,
        scratch.d_q,
        ctx.batch_head,
        ctx.seq_len,
        ctx.head_dim,
        ctx.seq_len,
    )?;
    run_tc_matmul(
        ctx.stream,
        ctx.tc_module,
        scratch.ds_t,
        scratch.q_t,
        scratch.d_k,
        ctx.batch_head,
        ctx.seq_len,
        ctx.head_dim,
        ctx.seq_len,
    )?;
    run_tc_matmul(
        ctx.stream,
        ctx.tc_module,
        scratch.p_t,
        scratch.d_out_t,
        scratch.d_v,
        ctx.batch_head,
        ctx.seq_len,
        ctx.head_dim,
        ctx.seq_len,
    )
}
