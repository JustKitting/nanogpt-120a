use cuda_device::{DisjointSlice, SharedArray, thread};

use super::{CompactTileCtx, KdaChunkTileCtx};
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::f16_tc_matmul::cta_stage::{load_a_fragments, load_b_fragments};
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, CTA_THREADS, CtaTile};
use crate::kda_common::{chunk_state_index, compact_index};
use crate::mma::mma_m16n8k16_f16_f16_f32;

macro_rules! stage_compact_fn {
    ($name:ident, $src:ident: $src_ty:ty, $tile:ident: $tile_elems:ident,
     $row:ident, $col:ident, $ctx:ident, $k_base:ident, $value:expr) => {
        pub(crate) fn $name(
            $src: $src_ty,
            $tile: &mut SharedArray<u16, $tile_elems>,
            $ctx: CompactTileCtx<'_>,
            $k_base: u32,
        ) {
            let mut offset = thread::threadIdx_x();
            while offset < $tile_elems as u32 {
                let $row = offset / CTA_K;
                let $col = offset - $row * CTA_K;
                $tile[offset as usize] = $value;
                offset += CTA_THREADS;
            }
        }
    };
}

stage_compact_fn!(stage_compact_a, src: &[f32], a_tile: CTA_A_ELEMS, row, col, ctx, k_base, {
    let token = ctx.start + ctx.tile.row_base + row;
    let dim = k_base + col;
    if token < ctx.end && dim < ctx.params.head_dim {
        cvt_rn_f16_f32(src[compact_index(ctx.batch, token, ctx.head, dim, ctx.params)])
    } else {
        0
    }
});

stage_compact_fn!(stage_compact_b_t, src: &[f32], b_tile: CTA_B_ELEMS, row, col, ctx, k_base, {
    let v_dim = ctx.tile.col_base + row;
    let token = ctx.start + k_base + col;
    if v_dim < ctx.params.head_dim && token < ctx.end {
        cvt_rn_f16_f32(src[compact_index(ctx.batch, token, ctx.head, v_dim, ctx.params)])
    } else {
        0
    }
});

stage_compact_fn!(
    stage_compact_token_dim_b_t, src: &[f32], b_tile: CTA_B_ELEMS, row, col, ctx, k_base, {
        let token = ctx.start + ctx.tile.col_base + row;
        let dim = k_base + col;
        if token < ctx.end && dim < ctx.params.head_dim {
            cvt_rn_f16_f32(src[compact_index(ctx.batch, token, ctx.head, dim, ctx.params)])
        } else {
            0
        }
    }
);

stage_compact_fn!(
    stage_compact_b_t_disjoint, src: &mut DisjointSlice<f32>, b_tile: CTA_B_ELEMS, row, col, ctx,
    k_base, {
        let v_dim = ctx.tile.col_base + row;
        let token = ctx.start + k_base + col;
        if v_dim < ctx.params.head_dim && token < ctx.end {
            let index = compact_index(ctx.batch, token, ctx.head, v_dim, ctx.params);
            cvt_rn_f16_f32(unsafe { *src.get_unchecked_mut(index) })
        } else {
            0
        }
    }
);

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

#[inline(always)]
pub(crate) fn mma_accumulate(
    tile: CtaTile,
    a_tile: &SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &SharedArray<u16, CTA_B_ELEMS>,
    acc: &mut [[f32; 4]; 4],
) {
    let a_fragments = load_a_fragments(a_tile, tile);
    let mut i = 0;
    while i < 4 {
        mma_m16n8k16_f16_f16_f32(
            a_fragments,
            load_b_fragments(b_tile, tile, tile.warp_n0 + i as u32),
            &mut acc[i],
        );
        i += 1;
    }
}
