use cuda_device::DisjointSlice;

use crate::mma::projection::Nvfp4ProjectionParams;

use super::super::tile::Nvfp4ProjectionCtaTile;
use super::common::{affine_value, affine_value_scaled, row_col};

#[inline(always)]
pub fn store_affine_accumulator(
    acc: [f32; 4],
    input_global_scales: &[f32],
    bias_bytes: &[u8],
    bias_scales: &[u8],
    out: &mut DisjointSlice<'_, f32>,
    tile: Nvfp4ProjectionCtaTile,
    params: &Nvfp4ProjectionParams,
) {
    store_one(
        acc[0],
        0,
        input_global_scales,
        bias_bytes,
        bias_scales,
        out,
        tile,
        params,
    );
    store_one(
        acc[1],
        1,
        input_global_scales,
        bias_bytes,
        bias_scales,
        out,
        tile,
        params,
    );
    store_one(
        acc[2],
        2,
        input_global_scales,
        bias_bytes,
        bias_scales,
        out,
        tile,
        params,
    );
    store_one(
        acc[3],
        3,
        input_global_scales,
        bias_bytes,
        bias_scales,
        out,
        tile,
        params,
    );
}

#[inline(always)]
pub fn store_affine_accumulator_aligned(
    acc: [f32; 4],
    input_global_scales: &[f32],
    bias_bytes: &[u8],
    bias_scales: &[u8],
    out: &mut DisjointSlice<'_, f32>,
    tile: Nvfp4ProjectionCtaTile,
    params: &Nvfp4ProjectionParams,
) {
    let row0 = tile.mma_row_base() + tile.group;
    let row1 = row0 + 8;
    let col0 = tile.mma_col_base() + tile.thread_in_group * 2;
    let scale0 = input_global_scales[row0 as usize] * params.weight_global_scale;
    let scale1 = input_global_scales[row1 as usize] * params.weight_global_scale;
    store_pair_aligned(
        acc[0],
        acc[1],
        row0,
        col0,
        scale0,
        bias_bytes,
        bias_scales,
        out,
        params,
    );
    store_pair_aligned(
        acc[2],
        acc[3],
        row1,
        col0,
        scale1,
        bias_bytes,
        bias_scales,
        out,
        params,
    );
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
fn store_one(
    acc: f32,
    index: u32,
    input_global_scales: &[f32],
    bias_bytes: &[u8],
    bias_scales: &[u8],
    out: &mut DisjointSlice<'_, f32>,
    tile: Nvfp4ProjectionCtaTile,
    params: &Nvfp4ProjectionParams,
) {
    let (row, col) = row_col(tile, index);
    if row < params.token_count && col < params.output_dim {
        let value = affine_value(
            acc,
            row,
            col,
            input_global_scales,
            bias_bytes,
            bias_scales,
            params,
        );
        let offset = row as usize * params.output_dim as usize + col as usize;
        unsafe {
            *out.get_unchecked_mut(offset) = if params.residual_add == 0 {
                value
            } else {
                *out.get_unchecked_mut(offset) + value
            };
        }
    }
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
fn store_pair_aligned(
    acc0: f32,
    acc1: f32,
    row: u32,
    col0: u32,
    scale: f32,
    bias_bytes: &[u8],
    bias_scales: &[u8],
    out: &mut DisjointSlice<'_, f32>,
    params: &Nvfp4ProjectionParams,
) {
    let value0 = affine_value_scaled(acc0, scale, col0, bias_bytes, bias_scales, params);
    let value1 = affine_value_scaled(acc1, scale, col0 + 1, bias_bytes, bias_scales, params);
    let offset = row as usize * params.output_dim as usize + col0 as usize;
    unsafe {
        if params.residual_add == 0 {
            *out.get_unchecked_mut(offset) = value0;
            *out.get_unchecked_mut(offset + 1) = value1;
        } else {
            *out.get_unchecked_mut(offset) += value0;
            *out.get_unchecked_mut(offset + 1) += value1;
        }
    }
}
