use cuda_device::{DisjointSlice, SharedArray};

use super::{CompactTileCtx, MatrixTileCtx};
use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::cta_tile::CtaTile;
use crate::kda_common::{chunk_matrix_index, compact_index, hidden_index};

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
