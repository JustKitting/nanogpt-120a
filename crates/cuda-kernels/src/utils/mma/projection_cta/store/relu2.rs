use cuda_device::DisjointSlice;

use crate::float_ptx::max_f32;
use crate::mma::projection::Nvfp4ProjectionParams;

use super::super::tile::Nvfp4ProjectionCtaTile;
use super::common::{affine_value, affine_value_scaled, row_col};

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn store_relu2_accumulator(
    acc: [f32; 4],
    input_global_scales: &[f32],
    bias_bytes: &[u8],
    bias_scales: &[u8],
    pre_activation: &mut DisjointSlice<'_, f32>,
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
        pre_activation,
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
        pre_activation,
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
        pre_activation,
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
        pre_activation,
        out,
        tile,
        params,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn store_relu2_accumulator_aligned(
    acc: [f32; 4],
    input_global_scales: &[f32],
    bias_bytes: &[u8],
    bias_scales: &[u8],
    pre_activation: &mut DisjointSlice<'_, f32>,
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
        pre_activation,
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
        pre_activation,
        out,
        params,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn store_one(
    acc: f32,
    index: u32,
    input_global_scales: &[f32],
    bias_bytes: &[u8],
    bias_scales: &[u8],
    pre_activation: &mut DisjointSlice<'_, f32>,
    out: &mut DisjointSlice<'_, f32>,
    tile: Nvfp4ProjectionCtaTile,
    params: &Nvfp4ProjectionParams,
) {
    let (row, col) = row_col(tile, index);
    if row < params.token_count && col < params.output_dim {
        let pre = affine_value(
            acc,
            row,
            col,
            input_global_scales,
            bias_bytes,
            bias_scales,
            params,
        );
        let relu = max_f32(pre, 0.0);
        let offset = row as usize * params.output_dim as usize + col as usize;
        unsafe {
            *pre_activation.get_unchecked_mut(offset) = pre;
            *out.get_unchecked_mut(offset) = relu * relu;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn store_pair_aligned(
    acc0: f32,
    acc1: f32,
    row: u32,
    col0: u32,
    scale: f32,
    bias_bytes: &[u8],
    bias_scales: &[u8],
    pre_activation: &mut DisjointSlice<'_, f32>,
    out: &mut DisjointSlice<'_, f32>,
    params: &Nvfp4ProjectionParams,
) {
    let pre0 = affine_value_scaled(acc0, scale, col0, bias_bytes, bias_scales, params);
    let pre1 = affine_value_scaled(acc1, scale, col0 + 1, bias_bytes, bias_scales, params);
    let relu0 = max_f32(pre0, 0.0);
    let relu1 = max_f32(pre1, 0.0);
    let offset = row as usize * params.output_dim as usize + col0 as usize;
    unsafe {
        *pre_activation.get_unchecked_mut(offset) = pre0;
        *pre_activation.get_unchecked_mut(offset + 1) = pre1;
        *out.get_unchecked_mut(offset) = relu0 * relu0;
        *out.get_unchecked_mut(offset + 1) = relu1 * relu1;
    }
}
