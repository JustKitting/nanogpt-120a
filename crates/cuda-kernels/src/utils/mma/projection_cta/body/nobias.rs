#![expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]

use cuda_device::{DisjointSlice, SharedArray, thread};

use crate::mma::projection::Nvfp4ProjectionParams;
use crate::mma::projection_cta::accumulate::{
    projection_accumulator, projection_accumulator_aligned, projection_accumulator_aligned_row_pair,
};
use crate::mma::projection_cta::store::{store_accumulator, store_accumulator_aligned};
use crate::mma::projection_cta::tile::{
    NVFP4_PROJECTION_CTA_A_PACKS, NVFP4_PROJECTION_CTA_A_SCALES, NVFP4_PROJECTION_CTA_B_PACKS,
    NVFP4_PROJECTION_CTA_B_SCALES, NVFP4_PROJECTION_CTA_THREADS, Nvfp4ProjectionCtaTile,
};

macro_rules! nobias_body_at_fn {
    ($name:ident, $accumulator:ident, $store:ident) => {
        pub fn $name(
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
            let acc = $accumulator(
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

            $store(acc, input_global_scales, out, tile, &params);
        }
    };
}

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

nobias_body_at_fn!(
    nvfp4_projection_cta_nobias_kernel_body_at,
    projection_accumulator,
    store_accumulator
);
nobias_body_at_fn!(
    nvfp4_projection_cta_nobias_kernel_body_at_aligned,
    projection_accumulator_aligned,
    store_accumulator_aligned
);

pub fn nvfp4_projection_cta_nobias_kernel_body_at_aligned_row_pair(
    input_bytes: &[u8],
    input_scales: &[u8],
    input_global_scales: &[f32],
    weight_bytes: &[u8],
    weight_scales: &[u8],
    out: &mut DisjointSlice<'_, f32>,
    params: Nvfp4ProjectionParams,
    a_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS>,
    a1_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS>,
    b_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS>,
    a_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES>,
    a1_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES>,
    b_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES>,
    tile0: Nvfp4ProjectionCtaTile,
    tile1: Nvfp4ProjectionCtaTile,
) {
    let (acc0, acc1) = projection_accumulator_aligned_row_pair(
        input_bytes,
        input_scales,
        weight_bytes,
        weight_scales,
        tile0,
        tile1,
        &params,
        a_packs,
        a1_packs,
        b_packs,
        a_scales,
        a1_scales,
        b_scales,
    );

    store_accumulator_aligned(acc0, input_global_scales, out, tile0, &params);
    if tile1.row_base < params.token_count {
        store_accumulator_aligned(acc1, input_global_scales, out, tile1, &params);
    }
}
