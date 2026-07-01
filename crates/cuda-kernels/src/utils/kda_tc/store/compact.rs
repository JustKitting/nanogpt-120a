use cuda_device::DisjointSlice;

use super::coords::compact_fragment_coords;
use crate::kda_common::{compact_index, hidden_index};
use crate::kda_tc::CompactTileCtx;

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
