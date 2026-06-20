use cuda_device::DisjointSlice;

use crate::float_ptx::max_f32;
use crate::mma::projection::Nvfp4ProjectionParams;

use super::super::tile::Nvfp4ProjectionCtaTile;
use super::common::{affine_value, row_col};

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
