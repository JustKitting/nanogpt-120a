use cuda_device::DisjointSlice;

use super::{StoreAccumulatorNoBiasArgs, row_col};
use crate::mma::Nvfp4ProjectionTile;

#[inline(always)]
pub(crate) fn store_accumulator_nobias(
    acc: [f32; 4],
    tile: Nvfp4ProjectionTile,
    args: StoreAccumulatorNoBiasArgs<'_>,
    out: &mut DisjointSlice<'_, f32>,
) {
    store_one(acc[0], 0, tile, &args, out);
    store_one(acc[1], 1, tile, &args, out);
    store_one(acc[2], 2, tile, &args, out);
    store_one(acc[3], 3, tile, &args, out);
}

#[inline(always)]
fn store_one(
    acc: f32,
    index: usize,
    tile: Nvfp4ProjectionTile,
    args: &StoreAccumulatorNoBiasArgs<'_>,
    out: &mut DisjointSlice<'_, f32>,
) {
    let (row, col) = row_col(tile, index);
    if row < args.params.token_count && col < args.params.output_dim {
        let global_scale = args.input_global_scales[row as usize] * args.params.weight_global_scale;
        let offset = row as usize * args.params.output_dim as usize + col as usize;
        unsafe {
            *out.get_unchecked_mut(offset) = acc * global_scale;
        }
    }
}
