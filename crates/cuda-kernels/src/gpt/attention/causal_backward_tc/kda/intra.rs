use cuda_device::{DisjointSlice, thread};

use super::super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use crate::attention::CausalAttentionParams;
use crate::float_ptx::fma_f32;
use crate::kda_common::{beta_compact_index, compact_index, kda_decay_exp, safe_denom};
use crate::kda_tc::KdaChunkTileCtx;

#[derive(Clone, Copy)]
pub(crate) struct KdaIntraInputs<'a> { pub(crate) qg: &'a [f32], pub(crate) kg: &'a [f32], pub(crate) vbeta: &'a [f32], pub(crate) g: &'a [f32], pub(crate) beta: &'a [f32], pub(crate) d_kg: &'a [f32], pub(crate) d_kpos_m: &'a [f32], pub(crate) d_vbeta_m: &'a [f32], pub(crate) d_kneg_from_b: &'a [f32], pub(crate) d_kpos_from_b_t: &'a [f32] }

pub(crate) struct KdaIntraGrads<'a> { pub(crate) qg_to_dv: DisjointSlice<'a, f32>, pub(crate) k_a_to_dg: DisjointSlice<'a, f32>, pub(crate) q: DisjointSlice<'a, f32>, pub(crate) k: DisjointSlice<'a, f32>, pub(crate) beta: DisjointSlice<'a, f32> }

#[derive(Clone, Copy)]
struct KdaIntraCtx<'a> { params: &'a CausalAttentionParams, batch: u32, head: u32, start: u32, end: u32, head_dim: u32, chunk_tokens: u32 }

impl KdaIntraCtx<'_> {
    fn compact(self, token: u32, dim: u32) -> usize { compact_index(self.batch, token, self.head, dim, self.params) }

    fn beta(self, token: u32) -> usize { beta_compact_index(self.batch, token, self.head, self.params) }

    fn last_compact(self, dim: u32) -> usize { self.compact(self.end - 1, dim) }
}

pub(crate) fn chunk_intra_kda_backward_body(
    inputs: KdaIntraInputs<'_>,
    mut grads: KdaIntraGrads<'_>,
    params: CausalAttentionParams,
) {
    let Some(ctx) = KdaChunkTileCtx::from_block(&params) else {
        return;
    };
    let tid = thread::threadIdx_x();
    let compact_ctx = ctx.compact;
    let ctx = KdaIntraCtx { params: &params, batch: compact_ctx.batch, head: compact_ctx.head, start: compact_ctx.start, end: compact_ctx.end, head_dim: params.head_dim, chunk_tokens: ctx.matrix.chunk_tokens };

    update_compact_grads(inputs, &mut grads, ctx, tid);
    thread::sync_threads();

    update_beta_grad(inputs, &mut grads.beta, ctx, tid);
    thread::sync_threads();

    reverse_prefix_dg(&mut grads.k_a_to_dg, ctx, tid);
}

fn update_compact_grads(inputs: KdaIntraInputs<'_>, grads: &mut KdaIntraGrads<'_>, ctx: KdaIntraCtx<'_>, tid: u32) {
    let mut idx = tid;
    while idx < ctx.params.chunk_size * ctx.head_dim {
        let token_in_chunk = idx / ctx.head_dim;
        let dim = idx - token_in_chunk * ctx.head_dim;
        let token = ctx.start + token_in_chunk;
        if token < ctx.end {
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
        idx += TC_BACKWARD_THREADS_PER_BLOCK;
    }
}

fn update_beta_grad(inputs: KdaIntraInputs<'_>, beta_grad: &mut DisjointSlice<f32>, ctx: KdaIntraCtx<'_>, tid: u32) {
    let token_lane = tid;
    if token_lane < ctx.chunk_tokens {
        let token = ctx.start + token_lane;
        let beta_value = inputs.beta[ctx.beta(token)];
        let mut db_value = 0.0;

        let mut dim = 0;
        while dim < ctx.head_dim {
            let compact = ctx.compact(token, dim);
            let g_value = inputs.g[compact];
            let g_last = inputs.g[ctx.last_compact(dim)];
            let k_value = inputs.kg[compact] * kda_decay_exp(g_value - g_last);
            let exp_g = kda_decay_exp(g_value);

            let d_kpos = inputs.d_kpos_m[compact] + inputs.d_kneg_from_b[compact];
            db_value = fma_f32(d_kpos, k_value * exp_g, db_value);
            dim += 1;
        }

        let mut v_dim = 0;
        while v_dim < ctx.head_dim {
            let v_compact = ctx.compact(token, v_dim);
            let v_value = inputs.vbeta[v_compact] / safe_denom(beta_value);
            let d_vbeta = inputs.d_vbeta_m[v_compact];
            db_value = fma_f32(d_vbeta, v_value, db_value);
            v_dim += 1;
        }

        unsafe {
            *beta_grad.get_unchecked_mut(ctx.beta(token)) = db_value;
        }
    }
}

fn reverse_prefix_dg(k_a_to_dg: &mut DisjointSlice<f32>, ctx: KdaIntraCtx<'_>, tid: u32) {
    let dim = tid;
    if dim < ctx.head_dim {
        let mut acc = 0.0;
        let mut token = ctx.end;
        while token > ctx.start {
            token -= 1;
            let compact = ctx.compact(token, dim);
            acc += unsafe { *k_a_to_dg.get_unchecked_mut(compact) };
            unsafe {
                *k_a_to_dg.get_unchecked_mut(compact) = acc;
            }
        }
    }
}
