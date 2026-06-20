use cuda_core::DriverError;

use super::matmul::{AttentionTcMatmulContext, run_tc_matmul_rhs};
use super::types::CausalAttentionBackwardTcScratch;

pub(super) fn run_grad_matmuls(
    ctx: &AttentionTcMatmulContext<'_>,
    scratch: &mut CausalAttentionBackwardTcScratch<'_>,
) -> Result<(), DriverError> {
    run_tc_matmul_rhs(
        ctx.stream,
        ctx.tc_module,
        scratch.ds,
        scratch.k,
        scratch.d_q,
        ctx.batch_head,
        ctx.seq_len,
        ctx.head_dim,
        ctx.seq_len,
    )?;
    run_tc_matmul_rhs(
        ctx.stream,
        ctx.tc_module,
        scratch.ds_t,
        scratch.q,
        scratch.d_k,
        ctx.batch_head,
        ctx.seq_len,
        ctx.head_dim,
        ctx.seq_len,
    )?;
    run_tc_matmul_rhs(
        ctx.stream,
        ctx.tc_module,
        scratch.p_t,
        scratch.d_out,
        scratch.d_v,
        ctx.batch_head,
        ctx.seq_len,
        ctx.head_dim,
        ctx.seq_len,
    )
}
