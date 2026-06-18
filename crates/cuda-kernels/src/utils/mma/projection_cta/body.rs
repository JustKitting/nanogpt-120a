use cuda_device::{DisjointSlice, SharedArray, thread};

use crate::mma::mma_m16n8k64_scale4x_ue4m3;
use crate::mma::projection::Nvfp4ProjectionParams;
use crate::mma::projection_cta::load::{
    load_a_fragments, load_a_scale4, load_b_fragments, load_b_scale4,
};
use crate::mma::projection_cta::stage::stage_tiles;
use crate::mma::projection_cta::store::store_accumulator;
use crate::mma::projection_cta::tile::{
    NVFP4_PROJECTION_CTA_A_PACKS, NVFP4_PROJECTION_CTA_A_SCALES, NVFP4_PROJECTION_CTA_B_PACKS,
    NVFP4_PROJECTION_CTA_B_SCALES, NVFP4_PROJECTION_CTA_K, NVFP4_PROJECTION_CTA_THREADS,
    Nvfp4ProjectionCtaTile,
};

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
    let mut acc = [0.0_f32; 4];
    let mut k_base = 0;

    while k_base < params.input_dim {
        stage_tiles(
            input_bytes,
            input_scales,
            weight_bytes,
            weight_scales,
            tile,
            k_base,
            &params,
            a_packs,
            b_packs,
            a_scales,
            b_scales,
        );
        thread::sync_threads();
        mma_m16n8k64_scale4x_ue4m3(
            load_a_fragments(a_packs, tile),
            load_b_fragments(b_packs, tile),
            &mut acc,
            load_a_scale4(a_scales, tile),
            load_b_scale4(b_scales, tile),
        );
        thread::sync_threads();
        k_base += NVFP4_PROJECTION_CTA_K;
    }

    store_accumulator(acc, input_global_scales, out, tile, &params);
}
