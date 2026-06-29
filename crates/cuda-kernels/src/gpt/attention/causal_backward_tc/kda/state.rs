use cuda_device::{DisjointSlice, SharedArray, thread};

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS};
use crate::kda_tc::{
    CompactStore::SetScaled, KdaChunkTileCtx, StateTileLayout, StateTileSource,
    stage_compact_a as stage_dm_compact_a, stage_state_b_t, store_compact_quads, store_vnew_quads,
    tc_stage_loop,
};

pub(crate) fn chunk_kda_dkg_from_vnew_dh_body(
    v_new: &[f32],
    d_h_states: &[f32],
    mut d_kg: DisjointSlice<f32>,
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
        stage_dm_compact_a(v_new, a_tile, compact_ctx, k_base);
    } {
        stage_state_b_t(StateTileSource::F32(d_h_states), b_tile, ctx, k_base, StateTileLayout::VK);
    });

    store_compact_quads(acc, &mut d_kg, compact_ctx, SetScaled(1.0));
}

#[derive(Clone, Copy)]
pub(crate) enum ChunkStateMatmulMode {
    VNew,
    Dw,
    Dqg,
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(crate) fn chunk_state_matmul_body(
    a_src: &[f32],
    vnew_base: &[f32],
    state_u16: &[u16],
    mut out: DisjointSlice<f32>,
    params: CausalAttentionParams,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    mode: ChunkStateMatmulMode,
) {
    let Some(ctx) = KdaChunkTileCtx::from_block(&params) else {
        return;
    };
    let compact_ctx = ctx.compact;

    let mut acc = [[0.0_f32; 4]; 4];
    tc_stage_loop!(compact_ctx.tile, a_tile, b_tile, acc; k_base < params.head_dim; {
        stage_dm_compact_a(a_src, a_tile, compact_ctx, k_base);
    } {
        match mode {
            ChunkStateMatmulMode::VNew => {
                stage_state_b_t(StateTileSource::F16(state_u16), b_tile, ctx, k_base, StateTileLayout::KV)
            }
            ChunkStateMatmulMode::Dw | ChunkStateMatmulMode::Dqg => {
                stage_state_b_t(StateTileSource::F16(state_u16), b_tile, ctx, k_base, StateTileLayout::VK)
            },
        }
    });

    match mode {
        ChunkStateMatmulMode::VNew => store_vnew_quads(acc, vnew_base, &mut out, compact_ctx),
        ChunkStateMatmulMode::Dw => {
            store_compact_quads(acc, &mut out, compact_ctx, SetScaled(-1.0))
        }
        ChunkStateMatmulMode::Dqg => {
            store_compact_quads(acc, &mut out, compact_ctx, SetScaled(1.0))
        }
    }
}
