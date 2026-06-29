use cuda_device::DisjointSlice;

use super::accumulate::nvfp4_projection_accumulate_tile;
use super::args::Nvfp4ProjectionParams;
use super::body::active_projection_tile;
use super::store::{StoreAccumulatorArgs, store_relu2_accumulator, store_residual_accumulator};

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub fn nvfp4_projection_relu2_kernel_body(
    input_bytes: &[u8],
    input_scales: &[u8],
    input_global_scales: &[f32],
    weight_bytes: &[u8],
    weight_scales: &[u8],
    bias_bytes: &[u8],
    bias_scales: &[u8],
    pre_activation: &mut DisjointSlice<'_, f32>,
    out: &mut DisjointSlice<'_, f32>,
    params: Nvfp4ProjectionParams,
) {
    let Some(tile) = active_projection_tile() else {
        return;
    };
    let acc = nvfp4_projection_accumulate_tile(
        input_bytes,
        input_scales,
        weight_bytes,
        weight_scales,
        tile,
        &params,
    );
    let args = StoreAccumulatorArgs {
        input_global_scales,
        bias_bytes,
        bias_scales,
        params: &params,
    };
    store_relu2_accumulator(acc, tile, args, pre_activation, out);
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub fn nvfp4_projection_residual_kernel_body(
    input_bytes: &[u8],
    input_scales: &[u8],
    input_global_scales: &[f32],
    weight_bytes: &[u8],
    weight_scales: &[u8],
    bias_bytes: &[u8],
    bias_scales: &[u8],
    residual: &mut DisjointSlice<'_, f32>,
    projection_out: &mut DisjointSlice<'_, f32>,
    params: Nvfp4ProjectionParams,
) {
    let Some(tile) = active_projection_tile() else {
        return;
    };
    let acc = nvfp4_projection_accumulate_tile(
        input_bytes,
        input_scales,
        weight_bytes,
        weight_scales,
        tile,
        &params,
    );
    let args = StoreAccumulatorArgs {
        input_global_scales,
        bias_bytes,
        bias_scales,
        params: &params,
    };
    store_residual_accumulator(acc, tile, args, residual, projection_out);
}
