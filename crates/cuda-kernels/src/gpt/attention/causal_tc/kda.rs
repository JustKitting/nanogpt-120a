use cuda_device::{DisjointSlice, SharedArray, thread};

use super::gather::TC_FORWARD_THREADS_PER_BLOCK;
pub(super) use super::kda_elementwise::{
    chunk_cumsum_g_body, make_kg_kpos_vbeta_body, make_kneg_from_kg_body, make_qg_kneg_body,
    mask_akk_body, mask_aqk_body, prepare_kda_body, solve_akk_inv_body, store_chunk_g_last_body,
    zero_f32_body,
};
use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, CTA_THREADS, CtaTile};
use crate::kda_common::{
    KDA_STATE_ELEMS, batch_head, chunk_count, chunk_g_last_index, chunk_matrix_index,
    compact_index, kda_decay_exp, kda_tc_shape, state_elems,
};
use crate::kda_tc::{
    CompactTileCtx, KdaChunkTileCtx, MatrixTileCtx, StateTileLayout, StateTileSource,
    add_shared_state_quads, stage_compact_a, stage_compact_b_t as stage_vnew_b_t_slice,
    stage_compact_b_t_disjoint as stage_vnew_b_t_disjoint, stage_shared_state_b_t, stage_state_b_t,
    store_hidden_output_quads, store_vnew_quads, tc_stage_loop,
};

pub(super) fn chunk_kda_state_save_body(
    kg: &[f32],
    mut v_new: DisjointSlice<f32>,
    w: &[f32],
    u: &[f32],
    chunk_g_last: &[f32],
    mut chunk_states: DisjointSlice<u16>,
    params: CausalAttentionParams,
    state: &mut SharedArray<f32, KDA_STATE_ELEMS>,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
) {
    let bh = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    if bh >= batch_head(&params) || !kda_tc_shape(&params) {
        return;
    }
    let batch = bh / params.head_count;
    let head = bh - batch * params.head_count;

    let state_elems = state_elems(&params);
    let mut linear = tid;
    while linear < state_elems {
        state[linear as usize] = 0.0;
        linear += TC_FORWARD_THREADS_PER_BLOCK;
    }
    thread::sync_threads();

    let chunks = chunk_count(&params);
    let tile = CtaTile::from_tile(tid, 0, 0, 0);
    let mut chunk = 0;
    while chunk < chunks {
        let start = chunk * params.chunk_size;
        let end = params.seq_len.min(start + params.chunk_size);
        let ctx = CompactTileCtx::new(tile, (batch, head), (start, end), &params);

        linear = tid;
        while linear < state_elems {
            let base = ((bh * chunks + chunk) * state_elems) as usize;
            unsafe {
                *chunk_states.get_unchecked_mut(base + linear as usize) =
                    cvt_rn_f16_f32(state[linear as usize]);
            }
            linear += TC_FORWARD_THREADS_PER_BLOCK;
        }
        thread::sync_threads();

        compute_ws_to_vnew(w, u, &mut v_new, state, a_tile, b_tile, ctx);
        thread::sync_threads();

        decay_state(state, chunk_g_last, bh, chunk, tid, ctx);
        thread::sync_threads();
        compute_kg_vnew_add_state(kg, &mut v_new, state, a_tile, b_tile, ctx);
        thread::sync_threads();

        chunk += 1;
    }
}

pub(super) fn chunk_kda_output_from_state_body(
    q: &[f32],
    v_new: &[f32],
    aqk: &[f32],
    mut out: DisjointSlice<f32>,
    chunk_states: &[u16],
    params: CausalAttentionParams,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
) {
    let Some(ctx) = KdaChunkTileCtx::from_block(&params) else {
        return;
    };
    let compact_ctx = ctx.compact;

    let mut acc = [[0.0_f32; 4]; 4];
    tc_stage_loop!(compact_ctx.tile, a_tile, b_tile, acc; k_base < params.head_dim; {
        stage_compact_a(q, a_tile, compact_ctx, k_base);
    } {
        stage_state_b_t(StateTileSource::F16(chunk_states), b_tile, ctx, k_base, StateTileLayout::KV);
    });
    tc_stage_loop!(compact_ctx.tile, a_tile, b_tile, acc; k_base < params.chunk_size; {
        stage_chunk_matrix_a(aqk, a_tile, ctx.matrix, k_base);
    } {
        stage_vnew_b_t_slice(v_new, b_tile, compact_ctx, k_base);
    });

    store_hidden_output_quads(acc, &mut out, compact_ctx);
}

fn compute_ws_to_vnew(
    w: &[f32],
    u: &[f32],
    v_new: &mut DisjointSlice<f32>,
    state: &SharedArray<f32, KDA_STATE_ELEMS>,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    ctx: CompactTileCtx<'_>,
) {
    let mut acc = [[0.0_f32; 4]; 4];
    tc_stage_loop!(ctx.tile, a_tile, b_tile, acc; k_base < ctx.params.head_dim; {
        stage_compact_a(w, a_tile, ctx, k_base);
    } {
        stage_shared_state_b_t(state, b_tile, ctx, k_base);
    });
    store_vnew_quads(acc, u, v_new, ctx);
}

fn compute_kg_vnew_add_state(
    k: &[f32],
    v_new: &mut DisjointSlice<f32>,
    state: &mut SharedArray<f32, KDA_STATE_ELEMS>,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    ctx: CompactTileCtx<'_>,
) {
    let mut acc = [[0.0_f32; 4]; 4];
    tc_stage_loop!(ctx.tile, a_tile, b_tile, acc; k_base < ctx.params.chunk_size; {
        stage_kg_t_a(k, a_tile, ctx, k_base);
    } {
        stage_vnew_b_t_disjoint(v_new, b_tile, ctx, k_base);
    });
    add_shared_state_quads(acc, ctx.tile, state, ctx.params);
}

fn decay_state(
    state: &mut SharedArray<f32, KDA_STATE_ELEMS>,
    chunk_g_last: &[f32],
    bh: u32,
    chunk: u32,
    tid: u32,
    ctx: CompactTileCtx<'_>,
) {
    let state_elems = state_elems(ctx.params);
    let mut linear = tid;
    while linear < state_elems {
        let k_dim = linear / ctx.params.head_dim;
        let g_last = chunk_g_last[chunk_g_last_index(bh, chunk, k_dim, ctx.params)];
        state[linear as usize] *= kda_decay_exp(g_last);
        linear += TC_FORWARD_THREADS_PER_BLOCK;
    }
}

fn stage_chunk_matrix_a(
    src: &[f32],
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    ctx: MatrixTileCtx<'_>,
    k_base: u32,
) {
    let mut offset = thread::threadIdx_x();
    while offset < CTA_A_ELEMS as u32 {
        let row = offset / CTA_K;
        let col = offset - row * CTA_K;
        let token_in_chunk = ctx.tile.row_base + row;
        let source = k_base + col;
        let valid = token_in_chunk < ctx.params.chunk_size && source < ctx.params.chunk_size;
        let index = chunk_matrix_index(ctx.bh, ctx.chunk, token_in_chunk, source, ctx.params);
        a_tile[offset as usize] = if valid { cvt_rn_f16_f32(src[index]) } else { 0 };
        offset += CTA_THREADS;
    }
}

fn stage_kg_t_a(
    src: &[f32],
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    ctx: CompactTileCtx<'_>,
    k_base: u32,
) {
    let mut offset = thread::threadIdx_x();
    while offset < CTA_A_ELEMS as u32 {
        let row = offset / CTA_K;
        let col = offset - row * CTA_K;
        let dim = ctx.tile.row_base + row;
        let token = ctx.start + k_base + col;
        a_tile[offset as usize] = if dim < ctx.params.head_dim && token < ctx.end {
            cvt_rn_f16_f32(src[compact_index(ctx.batch, token, ctx.head, dim, ctx.params)])
        } else {
            0
        };
        offset += CTA_THREADS;
    }
}
