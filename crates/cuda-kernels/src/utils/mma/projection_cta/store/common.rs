use crate::float_ptx::fma_f32;
use crate::mma::projection::Nvfp4ProjectionParams;
use crate::nvfp4::nvfp4_value;

use super::super::tile::Nvfp4ProjectionCtaTile;

#[inline(always)]
pub fn affine_value(
    acc: f32,
    row: u32,
    col: u32,
    input_global_scales: &[f32],
    bias_bytes: &[u8],
    bias_scales: &[u8],
    params: &Nvfp4ProjectionParams,
) -> f32 {
    let global_scale = input_global_scales[row as usize] * params.weight_global_scale;
    affine_value_scaled(acc, global_scale, col, bias_bytes, bias_scales, params)
}

#[inline(always)]
pub fn affine_value_scaled(
    acc: f32,
    global_scale: f32,
    col: u32,
    bias_bytes: &[u8],
    bias_scales: &[u8],
    params: &Nvfp4ProjectionParams,
) -> f32 {
    let bias = nvfp4_value(
        bias_bytes,
        bias_scales,
        params.bias_global_scale,
        col as usize,
    );
    fma_f32(acc, global_scale, bias)
}

#[inline(always)]
pub fn row_col(tile: Nvfp4ProjectionCtaTile, index: u32) -> (u32, u32) {
    let row = tile.mma_row_base() + tile.group + if index < 2 { 0 } else { 8 };
    let col = tile.mma_col_base() + tile.thread_in_group * 2 + (index & 1);
    (row, col)
}
