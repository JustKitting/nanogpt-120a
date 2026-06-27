use cuda_device::{DisjointSlice, SharedArray, thread};

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::f16_tc_matmul::cta_stage::{load_a_fragments, load_b_fragments};
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, CTA_THREADS, CtaTile};
use crate::kda_common::{
    batch_head, chunk_count, chunk_matrix_index, chunk_state_index, compact_index, hidden_index,
    kda_tc_shape,
};
use crate::mma::mma_m16n8k16_f16_f16_f32;

macro_rules! tc_stage_loop {
    ($tile:expr, $a_tile:expr, $b_tile:expr, $acc:expr; $k_base:ident < $limit:expr;
     $stage_a:block $stage_b:block) => {{
        let mut $k_base = 0;
        while $k_base < $limit {
            $stage_a
            $stage_b
            thread::sync_threads();
            $crate::kda_tc::mma_accumulate($tile, $a_tile, $b_tile, &mut $acc);
            thread::sync_threads();
            $k_base += $crate::f16_tc_matmul::cta_tile::CTA_K;
        }
    }};
}

pub(crate) use tc_stage_loop;

macro_rules! for_acc_fragments {
    ($acc:expr, $tile:expr, |$warp_n:ident, $frag:ident, $value:ident| $body:block) => {{
        let mut i = 0;
        while i < 4 {
            let $warp_n = $tile.warp_n0 + i as u32;
            let mut $frag = 0;
            while $frag < 4 {
                let $value = $acc[i][$frag];
                $body
                $frag += 1;
            }
            i += 1;
        }
    }};
}

pub(crate) use for_acc_fragments;

#[derive(Clone, Copy)]
pub(crate) struct CompactTileCtx<'a> {
    pub(crate) tile: CtaTile,
    pub(crate) batch: u32,
    pub(crate) head: u32,
    pub(crate) start: u32,
    pub(crate) end: u32,
    pub(crate) params: &'a CausalAttentionParams,
}

impl<'a> CompactTileCtx<'a> {
    pub(crate) fn new(
        tile: CtaTile,
        batch_head: (u32, u32),
        token_span: (u32, u32),
        params: &'a CausalAttentionParams,
    ) -> Self {
        let (batch, head) = batch_head;
        let (start, end) = token_span;
        Self {
            tile,
            batch,
            head,
            start,
            end,
            params,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MatrixTileCtx<'a> {
    pub(crate) tile: CtaTile,
    pub(crate) bh: u32,
    pub(crate) chunk: u32,
    pub(crate) chunk_tokens: u32,
    pub(crate) params: &'a CausalAttentionParams,
}

impl<'a> MatrixTileCtx<'a> {
    pub(crate) fn new(
        tile: CtaTile,
        batch_chunk: (u32, u32),
        chunk_tokens: u32,
        params: &'a CausalAttentionParams,
    ) -> Self {
        let (bh, chunk) = batch_chunk;
        Self {
            tile,
            bh,
            chunk,
            chunk_tokens,
            params,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct KdaChunkTileCtx<'a> {
    pub(crate) bh: u32,
    pub(crate) chunk: u32,
    pub(crate) compact: CompactTileCtx<'a>,
    pub(crate) matrix: MatrixTileCtx<'a>,
}

impl<'a> KdaChunkTileCtx<'a> {
    pub(crate) fn from_block(params: &'a CausalAttentionParams) -> Option<Self> {
        let bh = thread::blockIdx_x();
        let chunk = thread::blockIdx_y();
        let tid = thread::threadIdx_x();
        let chunks = chunk_count(params);
        if bh >= batch_head(params) || chunk >= chunks || !kda_tc_shape(params) {
            return None;
        }
        let batch = bh / params.head_count;
        let head = bh - batch * params.head_count;
        let start = chunk * params.chunk_size;
        let end = params.seq_len.min(start + params.chunk_size);
        let tile = CtaTile::from_tile(tid, 0, 0, 0);
        Some(Self {
            bh,
            chunk,
            compact: CompactTileCtx::new(tile, (batch, head), (start, end), params),
            matrix: MatrixTileCtx::new(tile, (bh, chunk), end - start, params),
        })
    }
}

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

pub(crate) fn store_vnew_quads(
    acc: [[f32; 4]; 4],
    u: &[f32],
    v_new: &mut DisjointSlice<f32>,
    ctx: CompactTileCtx<'_>,
) {
    for_acc_fragments!(acc, ctx.tile, |warp_n, frag, value| {
        let (token_in_chunk, v_dim) = compact_fragment_coords(ctx.tile, warp_n, frag);
        let token = ctx.start + token_in_chunk;
        if token < ctx.end && v_dim < ctx.params.head_dim {
            let index = compact_index(ctx.batch, token, ctx.head, v_dim, ctx.params);
            unsafe {
                *v_new.get_unchecked_mut(index) = u[index] - value;
            }
        }
    });
}

#[derive(Clone, Copy)]
pub(crate) enum CompactStore {
    SetScaled(f32),
    Add,
}

pub(crate) fn store_compact_quads(
    acc: [[f32; 4]; 4],
    dst: &mut DisjointSlice<f32>,
    ctx: CompactTileCtx<'_>,
    mode: CompactStore,
) {
    for_acc_fragments!(acc, ctx.tile, |warp_n, frag, value| {
        let (token_in_chunk, dim) = compact_fragment_coords(ctx.tile, warp_n, frag);
        let token = ctx.start + token_in_chunk;
        if token < ctx.end && dim < ctx.params.head_dim {
            let index = compact_index(ctx.batch, token, ctx.head, dim, ctx.params);
            unsafe {
                match mode {
                    CompactStore::SetScaled(scale) => {
                        *dst.get_unchecked_mut(index) = value * scale;
                    }
                    CompactStore::Add => *dst.get_unchecked_mut(index) += value,
                }
            }
        }
    });
}

pub(crate) fn add_shared_state_quads<const STATE_ELEMS: usize>(
    acc: [[f32; 4]; 4],
    tile: CtaTile,
    state: &mut SharedArray<f32, STATE_ELEMS>,
    params: &CausalAttentionParams,
) {
    for_acc_fragments!(acc, tile, |warp_n, frag, value| {
        let (k_dim, v_dim) = compact_fragment_coords(tile, warp_n, frag);
        if k_dim < params.head_dim && v_dim < params.head_dim {
            state[(k_dim * params.head_dim + v_dim) as usize] += value;
        }
    });
}

pub(crate) fn store_hidden_output_quads(
    acc: [[f32; 4]; 4],
    out: &mut DisjointSlice<f32>,
    ctx: CompactTileCtx<'_>,
) {
    for_acc_fragments!(acc, ctx.tile, |warp_n, frag, value| {
        let (token_in_chunk, dim) = compact_fragment_coords(ctx.tile, warp_n, frag);
        let token = ctx.start + token_in_chunk;
        let row = ctx.batch * ctx.params.seq_len + token;
        if token < ctx.end && row < ctx.params.row_count && dim < ctx.params.head_dim {
            let index = hidden_index(ctx.batch, token, ctx.head, dim, ctx.params);
            unsafe {
                *out.get_unchecked_mut(index) = value;
            }
        }
    });
}

pub(crate) fn store_chunk_matrix_quads(
    acc: [[f32; 4]; 4],
    dst: &mut DisjointSlice<f32>,
    ctx: MatrixTileCtx<'_>,
) {
    for_acc_fragments!(acc, ctx.tile, |warp_n, frag, value| {
        let (row, col) = compact_fragment_coords(ctx.tile, warp_n, frag);
        if row < ctx.params.chunk_size && col < ctx.params.chunk_size {
            let matrix_value = if row < ctx.chunk_tokens && col < ctx.chunk_tokens {
                value
            } else {
                0.0
            };
            let index = chunk_matrix_index(ctx.bh, ctx.chunk, row, col, ctx.params);
            unsafe {
                *dst.get_unchecked_mut(index) = matrix_value;
            }
        }
    });
}

#[inline(always)]
pub(crate) fn compact_fragment_coords(tile: CtaTile, warp_n: u32, acc_index: usize) -> (u32, u32) {
    let token_in_chunk =
        tile.row_base + tile.warp_m * 16 + tile.group + if acc_index < 2 { 0 } else { 8 };
    let dim = tile.col_base + warp_n * 8 + tile.thread_in_group * 2 + (acc_index as u32 & 1);
    (token_in_chunk, dim)
}
