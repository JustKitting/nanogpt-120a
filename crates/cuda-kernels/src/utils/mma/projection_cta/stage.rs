use cuda_device::{SharedArray, thread};

use crate::mma::projection::Nvfp4ProjectionParams;

#[path = "stage/load.rs"]
mod load;
use load::{
    load_a_pack, load_a_pack_aligned, load_a_scale, load_a_scale_aligned, load_b_pack,
    load_b_pack_aligned, load_b_scale, load_b_scale_aligned,
};

use super::tile::{
    NVFP4_PROJECTION_CTA_A_PACKS, NVFP4_PROJECTION_CTA_A_SCALES, NVFP4_PROJECTION_CTA_B_PACKS,
    NVFP4_PROJECTION_CTA_B_SCALES, NVFP4_PROJECTION_CTA_THREADS, Nvfp4ProjectionCtaTile,
};

macro_rules! stage_tiles_fn {
    ($name:ident, $load_a_pack:ident, $load_b_pack:ident, $load_a_scale:ident, $load_b_scale:ident) => {
        pub fn $name(
            input_bytes: &[u8],
            input_scales: &[u8],
            weight_bytes: &[u8],
            weight_scales: &[u8],
            tile: Nvfp4ProjectionCtaTile,
            k_base: u32,
            params: &Nvfp4ProjectionParams,
            a_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS>,
            b_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS>,
            a_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES>,
            b_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES>,
        ) {
            let thread_id = thread::threadIdx_x();
            let mut offset = thread_id;
            while offset < NVFP4_PROJECTION_CTA_A_PACKS as u32 {
                a_packs[offset as usize] = $load_a_pack(input_bytes, tile, offset, k_base, params);
                offset += NVFP4_PROJECTION_CTA_THREADS;
            }

            let mut offset = thread_id;
            while offset < NVFP4_PROJECTION_CTA_B_PACKS as u32 {
                b_packs[offset as usize] = $load_b_pack(weight_bytes, tile, offset, k_base, params);
                offset += NVFP4_PROJECTION_CTA_THREADS;
            }

            let mut offset = thread_id;
            while offset < NVFP4_PROJECTION_CTA_A_SCALES as u32 {
                a_scales[offset as usize] =
                    $load_a_scale(input_scales, tile, offset, k_base, params);
                offset += NVFP4_PROJECTION_CTA_THREADS;
            }

            let mut offset = thread_id;
            while offset < NVFP4_PROJECTION_CTA_B_SCALES as u32 {
                b_scales[offset as usize] =
                    $load_b_scale(weight_scales, tile, offset, k_base, params);
                offset += NVFP4_PROJECTION_CTA_THREADS;
            }
        }
    };
}

stage_tiles_fn!(
    stage_tiles,
    load_a_pack,
    load_b_pack,
    load_a_scale,
    load_b_scale
);
stage_tiles_fn!(
    stage_tiles_aligned,
    load_a_pack_aligned,
    load_b_pack_aligned,
    load_a_scale_aligned,
    load_b_scale_aligned
);

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub fn stage_row_pair_tiles_aligned(
    input_bytes: &[u8],
    input_scales: &[u8],
    weight_bytes: &[u8],
    weight_scales: &[u8],
    tile0: Nvfp4ProjectionCtaTile,
    tile1: Nvfp4ProjectionCtaTile,
    k_base: u32,
    params: &Nvfp4ProjectionParams,
    a0_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS>,
    a1_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS>,
    b_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS>,
    a0_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES>,
    a1_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES>,
    b_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES>,
) {
    let thread_id = thread::threadIdx_x();
    let mut offset = thread_id;
    while offset < NVFP4_PROJECTION_CTA_B_PACKS as u32 {
        b_packs[offset as usize] = load_b_pack_aligned(weight_bytes, tile0, offset, k_base, params);
        if offset < NVFP4_PROJECTION_CTA_A_PACKS as u32 {
            a0_packs[offset as usize] =
                load_a_pack_aligned(input_bytes, tile0, offset, k_base, params);
            a1_packs[offset as usize] =
                load_a_pack_aligned(input_bytes, tile1, offset, k_base, params);
        }
        offset += NVFP4_PROJECTION_CTA_THREADS;
    }

    let mut offset = thread_id;
    while offset < NVFP4_PROJECTION_CTA_B_SCALES as u32 {
        b_scales[offset as usize] =
            load_b_scale_aligned(weight_scales, tile0, offset, k_base, params);
        if offset < NVFP4_PROJECTION_CTA_A_SCALES as u32 {
            a0_scales[offset as usize] =
                load_a_scale_aligned(input_scales, tile0, offset, k_base, params);
            a1_scales[offset as usize] =
                load_a_scale_aligned(input_scales, tile1, offset, k_base, params);
        }
        offset += NVFP4_PROJECTION_CTA_THREADS;
    }
}
