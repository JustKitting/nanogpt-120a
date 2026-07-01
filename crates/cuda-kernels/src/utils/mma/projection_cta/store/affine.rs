use cuda_device::DisjointSlice;

use crate::mma::projection::Nvfp4ProjectionParams;

use super::super::tile::Nvfp4ProjectionCtaTile;
use super::common::{affine_pair_scaled, affine_value, aligned_row_col, row_col};

struct AffineStoreArgs<'a> {
    input_global_scales: &'a [f32],
    bias_bytes: &'a [u8],
    bias_scales: &'a [u8],
    params: &'a Nvfp4ProjectionParams,
}

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
    let args = AffineStoreArgs {
        input_global_scales,
        bias_bytes,
        bias_scales,
        params,
    };
    store_acc4!(store_one, acc, out, tile, &args);
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
    let args = AffineStoreArgs {
        input_global_scales,
        bias_bytes,
        bias_scales,
        params,
    };
    let (row0, row1, col0) = aligned_row_col(tile);
    store_pair_aligned(acc[0], acc[1], row0, col0, out, &args);
    store_pair_aligned(acc[2], acc[3], row1, col0, out, &args);
}

#[inline(always)]
fn store_one(
    acc: f32,
    index: u32,
    out: &mut DisjointSlice<'_, f32>,
    tile: Nvfp4ProjectionCtaTile,
    args: &AffineStoreArgs<'_>,
) {
    let (row, col) = row_col(tile, index);
    if row < args.params.token_count && col < args.params.output_dim {
        let value = affine_value(
            acc,
            row,
            col,
            args.input_global_scales,
            args.bias_bytes,
            args.bias_scales,
            args.params,
        );
        let offset = row as usize * args.params.output_dim as usize + col as usize;
        unsafe {
            *out.get_unchecked_mut(offset) = if args.params.residual_add == 0 {
                value
            } else {
                *out.get_unchecked_mut(offset) + value
            };
        }
    }
}

#[inline(always)]
fn store_pair_aligned(
    acc0: f32,
    acc1: f32,
    row: u32,
    col0: u32,
    out: &mut DisjointSlice<'_, f32>,
    args: &AffineStoreArgs<'_>,
) {
    if row >= args.params.token_count || col0 + 1 >= args.params.output_dim {
        return;
    }
    let scale = args.input_global_scales[row as usize] * args.params.weight_global_scale;
    let (value0, value1) = affine_pair_scaled(
        (acc0, acc1),
        scale,
        col0,
        (args.bias_bytes, args.bias_scales),
        args.params,
    );
    let offset = row as usize * args.params.output_dim as usize + col0 as usize;
    unsafe {
        if args.params.residual_add == 0 {
            *out.get_unchecked_mut(offset) = value0;
            *out.get_unchecked_mut(offset + 1) = value1;
        } else {
            *out.get_unchecked_mut(offset) += value0;
            *out.get_unchecked_mut(offset + 1) += value1;
        }
    }
}
