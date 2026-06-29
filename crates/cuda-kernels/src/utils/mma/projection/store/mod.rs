mod affine;
mod nobias;

use crate::float_ptx::fma_f32;
use crate::nvfp4::nvfp4_value;

use super::args::Nvfp4ProjectionParams;
use super::args::Nvfp4ProjectionTile;

pub(super) use affine::{store_accumulator, store_relu2_accumulator, store_residual_accumulator};
pub(super) use nobias::store_accumulator_nobias;

pub(super) struct StoreAccumulatorArgs<'a> {
    pub(super) input_global_scales: &'a [f32],
    pub(super) bias_bytes: &'a [u8],
    pub(super) bias_scales: &'a [u8],
    pub(super) params: &'a Nvfp4ProjectionParams,
}

pub(super) struct StoreAccumulatorNoBiasArgs<'a> {
    pub(super) input_global_scales: &'a [f32],
    pub(super) params: &'a Nvfp4ProjectionParams,
}

#[inline(always)]
fn affine_value(acc: f32, row: u32, col: u32, args: &StoreAccumulatorArgs<'_>) -> f32 {
    let global_scale = args.input_global_scales[row as usize] * args.params.weight_global_scale;
    let bias = nvfp4_value(
        args.bias_bytes,
        args.bias_scales,
        args.params.bias_global_scale,
        col as usize,
    );
    fma_f32(acc, global_scale, bias)
}

#[inline(always)]
fn row_col(tile: Nvfp4ProjectionTile, acc_index: usize) -> (u32, u32) {
    let row = tile.tile_row + tile.group + if acc_index < 2 { 0 } else { 8 };
    let col = tile.tile_col + tile.thread_in_group * 2 + (acc_index as u32 & 1);
    (row, col)
}
