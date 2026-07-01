use super::super::super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use super::{KdaIntraCtx, KdaIntraGrads, KdaIntraInputs};
use crate::float_ptx::fma_f32;
use crate::kda_common::kda_decay_exp;

pub(super) fn update_compact_grads(
    inputs: KdaIntraInputs<'_>,
    grads: &mut KdaIntraGrads<'_>,
    ctx: KdaIntraCtx<'_>,
    tid: u32,
) {
    let mut idx = tid;
    while idx < ctx.params.chunk_size * ctx.head_dim {
        let token_in_chunk = idx / ctx.head_dim;
        let dim = idx - token_in_chunk * ctx.head_dim;
        let token = ctx.start + token_in_chunk;
        if token < ctx.end {
            update_compact_grad(inputs, grads, ctx, token, dim);
        }
        idx += TC_BACKWARD_THREADS_PER_BLOCK;
    }
}

fn update_compact_grad(
    inputs: KdaIntraInputs<'_>,
    grads: &mut KdaIntraGrads<'_>,
    ctx: KdaIntraCtx<'_>,
    token: u32,
    dim: u32,
) {
    let compact = ctx.compact(token, dim);
    let g_value = inputs.g[compact];
    let g_last = inputs.g[ctx.last_compact(dim)];
    let beta_value = inputs.beta[ctx.beta(token)];
    let exp_g = kda_decay_exp(g_value);
    let qg_value = inputs.qg[compact];
    let kg_value = inputs.kg[compact];
    let k_value = kg_value * kda_decay_exp(g_value - g_last);
    let kneg_value = k_value * kda_decay_exp(-g_value);
    let kpos_value = beta_value * k_value * exp_g;

    let d_qg_value = unsafe { *grads.qg_to_dv.get_unchecked_mut(compact) };
    let d_k_a_value = unsafe { *grads.k_a_to_dg.get_unchecked_mut(compact) };
    let d_kpos_m_value = inputs.d_kpos_m[compact];
    let d_kneg_b_value = inputs.d_kneg_from_b[compact];
    let d_kpos_b_t_value = inputs.d_kpos_from_b_t[compact];
    let dq_value = d_qg_value;
    let mut dk_value = inputs.d_kg[compact] * kda_decay_exp(g_last - g_value)
        + d_k_a_value * kda_decay_exp(-g_value)
        + d_kpos_m_value * beta_value * exp_g;
    let dv_value = inputs.d_vbeta_m[compact] * beta_value;
    let mut dg_value =
        -inputs.d_kg[compact] * kg_value - d_k_a_value * kneg_value + d_kpos_m_value * kpos_value;

    dk_value = fma_f32(d_kpos_b_t_value, kda_decay_exp(-g_value), dk_value);
    dk_value = fma_f32(d_kneg_b_value, beta_value * exp_g, dk_value);
    dg_value -= d_kpos_b_t_value * kneg_value;
    dg_value = fma_f32(d_kneg_b_value, kpos_value, dg_value);

    let mut kg_source = 0;
    let mut dg_last_value = 0.0;
    while kg_source < ctx.chunk_tokens {
        let source_token = ctx.start + kg_source;
        let source_compact = ctx.compact(source_token, dim);
        dg_last_value = fma_f32(inputs.d_kg[source_compact], inputs.kg[source_compact], dg_last_value);
        kg_source += 1;
    }
    if token == ctx.end - 1 {
        dg_value += dg_last_value;
    }

    unsafe {
        *grads.q.get_unchecked_mut(compact) = dq_value * exp_g;
        *grads.k.get_unchecked_mut(compact) = dk_value;
        *grads.qg_to_dv.get_unchecked_mut(compact) = dv_value;
        *grads.k_a_to_dg.get_unchecked_mut(compact) = dg_value + dq_value * qg_value;
    }
}
