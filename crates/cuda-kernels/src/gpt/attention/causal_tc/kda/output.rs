use cuda_device::{DisjointSlice, SharedArray, thread};

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, CTA_THREADS};
use crate::kda_common::chunk_matrix_index;
use crate::kda_tc::{
    KdaChunkTileCtx, MatrixTileCtx, StateTileLayout, StateTileSource, stage_compact_a,
    stage_compact_b_t as stage_vnew_b_t_slice, stage_state_b_t, store_hidden_output_quads,
    tc_stage_loop,
};

pub(in super::super) fn chunk_kda_output_from_state_body(
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
