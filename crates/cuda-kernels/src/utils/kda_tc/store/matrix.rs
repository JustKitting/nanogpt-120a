use cuda_device::DisjointSlice;

use super::coords::compact_fragment_coords;
use crate::kda_common::chunk_matrix_index;
use crate::kda_tc::MatrixTileCtx;

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
