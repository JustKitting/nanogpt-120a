use cuda_device::{DisjointSlice, thread};

use super::accumulate::nvfp4_projection_accumulate_tile;
use super::args::{NVFP4_PROJECTION_THREADS_PER_BLOCK, Nvfp4ProjectionParams};
use super::body::projection_tile;
use super::store::{StoreAccumulatorArgs, store_relu2_accumulator, store_residual_accumulator};

#[allow(clippy::too_many_arguments)]
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
    let lane = thread::threadIdx_x();
    if lane >= NVFP4_PROJECTION_THREADS_PER_BLOCK {
        return;
    }

    let tile = projection_tile(lane);
    let acc = nvfp4_projection_accumulate_tile(
        input_bytes,
        input_scales,
        weight_bytes,
        weight_scales,
        tile,
        &params,
    );
    let args =
        StoreAccumulatorArgs::new(input_global_scales, bias_bytes, bias_scales, tile, &params);
    store_relu2_accumulator(acc, tile, args, pre_activation, out);
}

#[allow(clippy::too_many_arguments)]
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
    let lane = thread::threadIdx_x();
    if lane >= NVFP4_PROJECTION_THREADS_PER_BLOCK {
        return;
    }

    let tile = projection_tile(lane);
    let acc = nvfp4_projection_accumulate_tile(
        input_bytes,
        input_scales,
        weight_bytes,
        weight_scales,
        tile,
        &params,
    );
    let args =
        StoreAccumulatorArgs::new(input_global_scales, bias_bytes, bias_scales, tile, &params);
    store_residual_accumulator(acc, tile, args, residual, projection_out);
}
