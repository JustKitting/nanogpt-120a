use cuda_device::{DisjointSlice, thread};

use super::accumulate::nvfp4_projection_accumulate_tile;
use super::args::{
    NVFP4_PROJECTION_M, NVFP4_PROJECTION_N, NVFP4_PROJECTION_THREADS_PER_BLOCK,
    Nvfp4ProjectionParams, Nvfp4ProjectionTile,
};
use super::store::{
    StoreAccumulatorArgs, StoreAccumulatorNoBiasArgs, store_accumulator, store_accumulator_nobias,
};

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub fn nvfp4_projection_kernel_body(
    input_bytes: &[u8],
    input_scales: &[u8],
    input_global_scales: &[f32],
    weight_bytes: &[u8],
    weight_scales: &[u8],
    bias_bytes: &[u8],
    bias_scales: &[u8],
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
    store_accumulator(
        acc,
        tile,
        StoreAccumulatorArgs::new(input_global_scales, bias_bytes, bias_scales, tile, &params),
        out,
    );
}

#[inline(always)]
pub fn nvfp4_projection_nobias_kernel_body(
    input_bytes: &[u8],
    input_scales: &[u8],
    input_global_scales: &[f32],
    weight_bytes: &[u8],
    weight_scales: &[u8],
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
    store_accumulator_nobias(
        acc,
        tile,
        StoreAccumulatorNoBiasArgs::new(input_global_scales, tile, &params),
        out,
    );
}

#[inline(always)]
pub(super) fn projection_tile(lane: u32) -> Nvfp4ProjectionTile {
    Nvfp4ProjectionTile {
        tile_col: thread::blockIdx_x() * NVFP4_PROJECTION_N,
        tile_row: thread::blockIdx_y() * NVFP4_PROJECTION_M,
        group: lane >> 2,
        thread_in_group: lane & 0x3,
    }
}
