use cuda_device::{DisjointSlice, SharedArray, thread};

use super::super::gather::TC_FORWARD_THREADS_PER_BLOCK;
use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, CTA_THREADS, CtaTile};
use crate::kda_common::{
    KDA_STATE_ELEMS, batch_head, chunk_count, chunk_g_last_index, compact_index, kda_decay_exp,
    kda_tc_shape, state_elems,
};
use crate::kda_tc::{
    CompactTileCtx, add_shared_state_quads, stage_compact_a,
    stage_compact_b_t_disjoint as stage_vnew_b_t_disjoint, stage_shared_state_b_t,
    store_vnew_quads, tc_stage_loop,
};

pub(in super::super) fn chunk_kda_state_save_body(
    kg: &[f32], mut v_new: DisjointSlice<f32>, w: &[f32], u: &[f32], chunk_g_last: &[f32],
    mut chunk_states: DisjointSlice<u16>,
    params: CausalAttentionParams,
    state: &mut SharedArray<f32, KDA_STATE_ELEMS>,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>, b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
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

fn compute_ws_to_vnew(
    w: &[f32], u: &[f32], v_new: &mut DisjointSlice<f32>,
    state: &SharedArray<f32, KDA_STATE_ELEMS>,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>, b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
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
    k: &[f32], v_new: &mut DisjointSlice<f32>,
    state: &mut SharedArray<f32, KDA_STATE_ELEMS>,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>, b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
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
    bh: u32, chunk: u32, tid: u32,
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

fn stage_kg_t_a(
    src: &[f32],
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    ctx: CompactTileCtx<'_>, k_base: u32,
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
