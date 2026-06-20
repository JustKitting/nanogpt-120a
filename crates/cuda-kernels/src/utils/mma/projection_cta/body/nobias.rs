use cuda_device::{DisjointSlice, SharedArray, thread};

use crate::mma::projection::Nvfp4ProjectionParams;
use crate::mma::projection_cta::accumulate::{
    projection_accumulator, projection_accumulator_aligned,
};
use crate::mma::projection_cta::store::{store_accumulator, store_accumulator_aligned};
use crate::mma::projection_cta::tile::{
    NVFP4_PROJECTION_CTA_A_PACKS, NVFP4_PROJECTION_CTA_A_SCALES, NVFP4_PROJECTION_CTA_B_PACKS,
    NVFP4_PROJECTION_CTA_B_SCALES, NVFP4_PROJECTION_CTA_THREADS, Nvfp4ProjectionCtaTile,
};

#[allow(clippy::too_many_arguments)]
pub fn nvfp4_projection_cta_nobias_kernel_body(
    input_bytes: &[u8],
    input_scales: &[u8],
    input_global_scales: &[f32],
    weight_bytes: &[u8],
    weight_scales: &[u8],
    out: &mut DisjointSlice<'_, f32>,
    params: Nvfp4ProjectionParams,
    a_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS>,
    b_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS>,
    a_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES>,
    b_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES>,
) {
    let thread_id = thread::threadIdx_x();
    if thread_id >= NVFP4_PROJECTION_CTA_THREADS {
        return;
    }

    let tile = Nvfp4ProjectionCtaTile::new(thread_id);
    nvfp4_projection_cta_nobias_kernel_body_at(
        input_bytes,
        input_scales,
        input_global_scales,
        weight_bytes,
        weight_scales,
        out,
        params,
        a_packs,
        b_packs,
        a_scales,
        b_scales,
        tile,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn nvfp4_projection_cta_nobias_kernel_body_at(
    input_bytes: &[u8],
    input_scales: &[u8],
    input_global_scales: &[f32],
    weight_bytes: &[u8],
    weight_scales: &[u8],
    out: &mut DisjointSlice<'_, f32>,
    params: Nvfp4ProjectionParams,
    a_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS>,
    b_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS>,
    a_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES>,
    b_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES>,
    tile: Nvfp4ProjectionCtaTile,
) {
    let acc = projection_accumulator(
        input_bytes,
        input_scales,
        weight_bytes,
        weight_scales,
        tile,
        &params,
        a_packs,
        b_packs,
        a_scales,
        b_scales,
    );

    store_accumulator(acc, input_global_scales, out, tile, &params);
}

#[allow(clippy::too_many_arguments)]
pub fn nvfp4_projection_cta_nobias_kernel_body_at_aligned(
    input_bytes: &[u8],
    input_scales: &[u8],
    input_global_scales: &[f32],
    weight_bytes: &[u8],
    weight_scales: &[u8],
    out: &mut DisjointSlice<'_, f32>,
    params: Nvfp4ProjectionParams,
    a_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS>,
    b_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS>,
    a_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES>,
    b_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES>,
    tile: Nvfp4ProjectionCtaTile,
) {
    let acc = projection_accumulator_aligned(
        input_bytes,
        input_scales,
        weight_bytes,
        weight_scales,
        tile,
        &params,
        a_packs,
        b_packs,
        a_scales,
        b_scales,
    );

    store_accumulator_aligned(acc, input_global_scales, out, tile, &params);
}
