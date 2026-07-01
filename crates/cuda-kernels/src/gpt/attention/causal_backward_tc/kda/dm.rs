use cuda_device::{DisjointSlice, thread};

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::f16_tc_matmul::cta_tile::{CTA_B_ELEMS, CTA_K, CTA_THREADS};
use crate::kda_common::{beta_compact_index, compact_index, kda_decay_exp};
use crate::kda_tc::{CompactTileCtx, CtaATile, CtaBTile, KdaChunkTileCtx, stage_compact_a as stage_dm_compact_a, stage_compact_token_dim_b_t as stage_dm_compact_b_t, store_chunk_matrix_quads, tc_stage_loop};

pub(crate) fn chunk_intra_kda_dm_body(
    kg: &[f32],
    vbeta: &[f32],
    g: &[f32],
    beta: &[f32],
    d_u: &[f32],
    d_w: &[f32],
    mut d_m: DisjointSlice<f32>,
    params: CausalAttentionParams,
    a_tile: &mut CtaATile,
    b_tile: &mut CtaBTile,
) {
    let Some(ctx) = KdaChunkTileCtx::from_block(&params) else {
        return;
    };
    let compact_ctx = ctx.compact;
    let mut acc = [[0.0_f32; 4]; 4];

    tc_stage_loop!(compact_ctx.tile, a_tile, b_tile, acc; k_base < params.head_dim; {
        stage_dm_compact_a(d_w, a_tile, compact_ctx, k_base);
    } {
        stage_dm_kpos_b_t(kg, g, beta, b_tile, compact_ctx, k_base);
    });

    tc_stage_loop!(compact_ctx.tile, a_tile, b_tile, acc; k_base < params.head_dim; {
        stage_dm_compact_a(d_u, a_tile, compact_ctx, k_base);
    } {
        stage_dm_compact_b_t(vbeta, b_tile, compact_ctx, k_base);
    });

    store_chunk_matrix_quads(acc, &mut d_m, ctx.matrix);
}

fn stage_dm_kpos_b_t(
    kg: &[f32],
    g: &[f32],
    beta: &[f32],
    b_tile: &mut CtaBTile,
    ctx: CompactTileCtx<'_>,
    k_base: u32,
) {
    let mut offset = thread::threadIdx_x();
    while offset < CTA_B_ELEMS as u32 {
        let row = offset / CTA_K;
        let col = offset - row * CTA_K;
        let source = ctx.tile.col_base + row;
        let dim = k_base + col;
        let token = ctx.start + source;
        b_tile[offset as usize] = if token < ctx.end && dim < ctx.params.head_dim {
            let compact = compact_index(ctx.batch, token, ctx.head, dim, ctx.params);
            let g_value = g[compact];
            let g_last = g[compact_index(ctx.batch, ctx.end - 1, ctx.head, dim, ctx.params)];
            let beta_value = beta[beta_compact_index(ctx.batch, token, ctx.head, ctx.params)];
            cvt_rn_f16_f32(beta_value * kg[compact] * kda_decay_exp(2.0 * g_value - g_last))
        } else {
            0
        };
        offset += CTA_THREADS;
    }
}
