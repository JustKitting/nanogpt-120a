use cuda_device::{DisjointSlice, thread};

use super::super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use crate::attention::CausalAttentionParams;
use crate::float_ptx::fma_f32;
use crate::kda_common::{beta_compact_index, compact_index, kda_decay_exp, safe_denom};
use crate::kda_tc::KdaChunkTileCtx;

pub(crate) fn chunk_intra_kda_backward_body(
    qg: &[f32],
    kg: &[f32],
    vbeta: &[f32],
    g: &[f32],
    beta: &[f32],
    mut d_qg_to_dv: DisjointSlice<f32>,
    d_kg: &[f32],
    mut d_k_a_to_dg: DisjointSlice<f32>,
    d_kpos_m: &[f32],
    d_vbeta_m: &[f32],
    d_kneg_from_b: &[f32],
    d_kpos_from_b_t: &[f32],
    mut d_q: DisjointSlice<f32>,
    mut d_k: DisjointSlice<f32>,
    mut d_beta: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let Some(ctx) = KdaChunkTileCtx::from_block(&params) else {
        return;
    };
    let tid = thread::threadIdx_x();
    let head_dim = params.head_dim;
    let compact_ctx = ctx.compact;
    let batch = compact_ctx.batch;
    let head = compact_ctx.head;
    let start = compact_ctx.start;
    let end = compact_ctx.end;
    let chunk_tokens = ctx.matrix.chunk_tokens;

    let mut idx = tid;
    while idx < params.chunk_size * head_dim {
        let token_in_chunk = idx / head_dim;
        let dim = idx - token_in_chunk * head_dim;
        let token = start + token_in_chunk;
        if token < end {
            let compact = compact_index(batch, token, head, dim, &params);
            let g_value = g[compact];
            let g_last = g[compact_index(batch, end - 1, head, dim, &params)];
            let beta_value = beta[beta_compact_index(batch, token, head, &params)];
            let exp_g = kda_decay_exp(g_value);
            let qg_value = qg[compact];
            let kg_value = kg[compact];
            let k_value = kg_value * kda_decay_exp(g_value - g_last);
            let kneg_value = k_value * kda_decay_exp(-g_value);
            let kpos_value = beta_value * k_value * exp_g;

            let d_qg_value = unsafe { *d_qg_to_dv.get_unchecked_mut(compact) };
            let d_k_a_value = unsafe { *d_k_a_to_dg.get_unchecked_mut(compact) };
            let d_kpos_m_value = d_kpos_m[compact];
            let d_kneg_b_value = d_kneg_from_b[compact];
            let d_kpos_b_t_value = d_kpos_from_b_t[compact];
            let dq_value = d_qg_value;
            let mut dk_value = d_kg[compact] * kda_decay_exp(g_last - g_value)
                + d_k_a_value * kda_decay_exp(-g_value)
                + d_kpos_m_value * beta_value * exp_g;
            let dv_value = d_vbeta_m[compact] * beta_value;
            let mut dg_value =
                -d_kg[compact] * kg_value - d_k_a_value * kneg_value + d_kpos_m_value * kpos_value;

            dk_value = fma_f32(d_kpos_b_t_value, kda_decay_exp(-g_value), dk_value);
            dk_value = fma_f32(d_kneg_b_value, beta_value * exp_g, dk_value);
            dg_value -= d_kpos_b_t_value * kneg_value;
            dg_value = fma_f32(d_kneg_b_value, kpos_value, dg_value);

            let mut kg_source = 0;
            let mut dg_last_value = 0.0;
            while kg_source < chunk_tokens {
                let source_token = start + kg_source;
                let source_compact = compact_index(batch, source_token, head, dim, &params);
                dg_last_value = fma_f32(d_kg[source_compact], kg[source_compact], dg_last_value);
                kg_source += 1;
            }
            if token == end - 1 {
                dg_value += dg_last_value;
            }

            unsafe {
                *d_q.get_unchecked_mut(compact) = dq_value * exp_g;
                *d_k.get_unchecked_mut(compact) = dk_value;
                *d_qg_to_dv.get_unchecked_mut(compact) = dv_value;
                *d_k_a_to_dg.get_unchecked_mut(compact) = dg_value + dq_value * qg_value;
            }
        }
        idx += TC_BACKWARD_THREADS_PER_BLOCK;
    }
    thread::sync_threads();

    let token_lane = tid;
    if token_lane < chunk_tokens {
        let token = start + token_lane;
        let beta_value = beta[beta_compact_index(batch, token, head, &params)];
        let mut db_value = 0.0;

        let mut dim = 0;
        while dim < head_dim {
            let compact = compact_index(batch, token, head, dim, &params);
            let g_value = g[compact];
            let g_last = g[compact_index(batch, end - 1, head, dim, &params)];
            let k_value = kg[compact] * kda_decay_exp(g_value - g_last);
            let exp_g = kda_decay_exp(g_value);

            let d_kpos = d_kpos_m[compact] + d_kneg_from_b[compact];
            db_value = fma_f32(d_kpos, k_value * exp_g, db_value);
            dim += 1;
        }

        let mut v_dim = 0;
        while v_dim < head_dim {
            let v_compact = compact_index(batch, token, head, v_dim, &params);
            let v_value = vbeta[v_compact] / safe_denom(beta_value);
            let d_vbeta = d_vbeta_m[v_compact];
            db_value = fma_f32(d_vbeta, v_value, db_value);
            v_dim += 1;
        }

        unsafe {
            *d_beta.get_unchecked_mut(beta_compact_index(batch, token, head, &params)) = db_value;
        }
    }
    thread::sync_threads();

    let dim = tid;
    if dim < head_dim {
        let mut acc = 0.0;
        let mut token = end;
        while token > start {
            token -= 1;
            let compact = compact_index(batch, token, head, dim, &params);
            acc += unsafe { *d_k_a_to_dg.get_unchecked_mut(compact) };
            unsafe {
                *d_k_a_to_dg.get_unchecked_mut(compact) = acc;
            }
        }
    }
}
