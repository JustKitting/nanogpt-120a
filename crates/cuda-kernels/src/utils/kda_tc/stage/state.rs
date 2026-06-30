use cuda_device::{SharedArray, thread};

use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::f16_tc_matmul::cta_tile::{CTA_B_ELEMS, CTA_K, CTA_THREADS};
use crate::kda_common::chunk_state_index;
use crate::kda_tc::{CompactTileCtx, KdaChunkTileCtx};

pub(crate) fn stage_shared_state_b_t<const STATE_ELEMS: usize>(
    state: &SharedArray<f32, STATE_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    ctx: CompactTileCtx<'_>,
    k_base: u32,
) {
    let mut offset = thread::threadIdx_x();
    while offset < CTA_B_ELEMS as u32 {
        let row = offset / CTA_K;
        let col = offset - row * CTA_K;
        let v_dim = ctx.tile.col_base + row;
        let k_dim = k_base + col;
        b_tile[offset as usize] = if v_dim < ctx.params.head_dim && k_dim < ctx.params.head_dim {
            cvt_rn_f16_f32(state[(k_dim * ctx.params.head_dim + v_dim) as usize])
        } else {
            0
        };
        offset += CTA_THREADS;
    }
}

#[derive(Clone, Copy)]
pub(crate) enum StateTileSource<'a> {
    F16(&'a [u16]),
    F32(&'a [f32]),
}

#[derive(Clone, Copy)]
pub(crate) enum StateTileLayout {
    KV,
    VK,
}

pub(crate) fn stage_state_b_t(
    source: StateTileSource<'_>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    ctx: KdaChunkTileCtx<'_>,
    k_base: u32,
    layout: StateTileLayout,
) {
    let mut offset = thread::threadIdx_x();
    while offset < CTA_B_ELEMS as u32 {
        let row = offset / CTA_K;
        let col = offset - row * CTA_K;
        let (k_dim, v_dim) = match layout {
            StateTileLayout::KV => (k_base + col, ctx.compact.tile.col_base + row),
            StateTileLayout::VK => (ctx.compact.tile.col_base + row, k_base + col),
        };
        b_tile[offset as usize] =
            if k_dim < ctx.compact.params.head_dim && v_dim < ctx.compact.params.head_dim {
                let index = chunk_state_index(
                    ctx.bh,
                    ctx.chunk,
                    k_dim * ctx.compact.params.head_dim + v_dim,
                    ctx.compact.params,
                );
                match source {
                    StateTileSource::F16(states) => states[index],
                    StateTileSource::F32(states) => cvt_rn_f16_f32(states[index]),
                }
            } else {
                0
            };
        offset += CTA_THREADS;
    }
}
