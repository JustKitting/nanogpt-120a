use cuda_device::DisjointSlice;

use crate::mma::projection::Nvfp4ProjectionParams;

use super::super::tile::Nvfp4ProjectionCtaTile;
use super::common::row_col;

#[inline(always)]
pub fn store_accumulator(
    acc: [f32; 4],
    input_global_scales: &[f32],
    out: &mut DisjointSlice<'_, f32>,
    tile: Nvfp4ProjectionCtaTile,
    params: &Nvfp4ProjectionParams,
) {
    store_one(acc[0], 0, input_global_scales, out, tile, params);
    store_one(acc[1], 1, input_global_scales, out, tile, params);
    store_one(acc[2], 2, input_global_scales, out, tile, params);
    store_one(acc[3], 3, input_global_scales, out, tile, params);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn store_one(
    acc: f32,
    index: u32,
    input_global_scales: &[f32],
    out: &mut DisjointSlice<'_, f32>,
    tile: Nvfp4ProjectionCtaTile,
    params: &Nvfp4ProjectionParams,
) {
    let (row, col) = row_col(tile, index);
    if row < params.token_count && col < params.output_dim {
        let global_scale = input_global_scales[row as usize] * params.weight_global_scale;
        let offset = row as usize * params.output_dim as usize + col as usize;
        unsafe {
            *out.get_unchecked_mut(offset) = acc * global_scale;
        }
    }
}
