use cuda_device::{DisjointSlice, SharedArray, thread};

use crate::mma::projection::Nvfp4ProjectionParams;
use crate::mma::projection_cta::accumulate::projection_accumulator;
use crate::mma::projection_cta::store::store_relu2_accumulator;
use crate::mma::projection_cta::tile::{
    NVFP4_PROJECTION_CTA_A_PACKS, NVFP4_PROJECTION_CTA_A_SCALES, NVFP4_PROJECTION_CTA_B_PACKS,
    NVFP4_PROJECTION_CTA_B_SCALES, NVFP4_PROJECTION_CTA_THREADS, Nvfp4ProjectionCtaTile,
};

#[allow(clippy::too_many_arguments)]
pub fn nvfp4_projection_cta_relu2_kernel_body(
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
    store_relu2_accumulator(
        acc,
        input_global_scales,
        bias_bytes,
        bias_scales,
        pre_activation,
        out,
        tile,
        &params,
    );
}
