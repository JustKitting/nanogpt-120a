use cuda_device::{SharedArray, thread};

use crate::mma::mma_m16n8k64_scale4x_ue4m3;
use crate::mma::projection::Nvfp4ProjectionParams;

use super::load::{load_a_fragments, load_a_scale4, load_b_fragments, load_b_scale4};
use super::stage::{
    stage_a_tiles_aligned, stage_b_tiles_aligned, stage_tiles, stage_tiles_aligned,
};
use super::tile::{
    NVFP4_PROJECTION_CTA_A_PACKS, NVFP4_PROJECTION_CTA_A_SCALES, NVFP4_PROJECTION_CTA_B_PACKS,
    NVFP4_PROJECTION_CTA_B_SCALES, NVFP4_PROJECTION_CTA_K, NVFP4_PROJECTION_CTA_K_ATOMS,
    Nvfp4ProjectionCtaTile,
};

#[allow(clippy::too_many_arguments)]
pub fn projection_accumulator(
    input_bytes: &[u8],
    input_scales: &[u8],
    weight_bytes: &[u8],
    weight_scales: &[u8],
    tile: Nvfp4ProjectionCtaTile,
    params: &Nvfp4ProjectionParams,
    a_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS>,
    b_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS>,
    a_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES>,
    b_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES>,
) -> [f32; 4] {
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
            params,
            a_packs,
            b_packs,
            a_scales,
            b_scales,
        );
        thread::sync_threads();
        let mut k_atom = 0;
        while k_atom < NVFP4_PROJECTION_CTA_K_ATOMS {
            mma_m16n8k64_scale4x_ue4m3(
                load_a_fragments(a_packs, tile, k_atom),
                load_b_fragments(b_packs, tile, k_atom),
                &mut acc,
                load_a_scale4(a_scales, tile, k_atom),
                load_b_scale4(b_scales, tile, k_atom),
            );
            k_atom += 1;
        }
        thread::sync_threads();
        k_base += NVFP4_PROJECTION_CTA_K;
    }

    acc
}

#[allow(clippy::too_many_arguments)]
pub fn projection_accumulator_aligned(
    input_bytes: &[u8],
    input_scales: &[u8],
    weight_bytes: &[u8],
    weight_scales: &[u8],
    tile: Nvfp4ProjectionCtaTile,
    params: &Nvfp4ProjectionParams,
    a_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS>,
    b_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS>,
    a_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES>,
    b_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES>,
) -> [f32; 4] {
    let mut acc = [0.0_f32; 4];
    let mut k_base = 0;

    while k_base < params.input_dim {
        stage_tiles_aligned(
            input_bytes,
            input_scales,
            weight_bytes,
            weight_scales,
            tile,
            k_base,
            params,
            a_packs,
            b_packs,
            a_scales,
            b_scales,
        );
        thread::sync_threads();
        let mut k_atom = 0;
        while k_atom < NVFP4_PROJECTION_CTA_K_ATOMS {
            mma_m16n8k64_scale4x_ue4m3(
                load_a_fragments(a_packs, tile, k_atom),
                load_b_fragments(b_packs, tile, k_atom),
                &mut acc,
                load_a_scale4(a_scales, tile, k_atom),
                load_b_scale4(b_scales, tile, k_atom),
            );
            k_atom += 1;
        }
        thread::sync_threads();
        k_base += NVFP4_PROJECTION_CTA_K;
    }

    acc
}

#[allow(clippy::too_many_arguments)]
pub fn projection_accumulator_aligned_row_pair(
    input_bytes: &[u8],
    input_scales: &[u8],
    weight_bytes: &[u8],
    weight_scales: &[u8],
    tile0: Nvfp4ProjectionCtaTile,
    tile1: Nvfp4ProjectionCtaTile,
    params: &Nvfp4ProjectionParams,
    a_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS>,
    a1_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS>,
    b_packs: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS>,
    a_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES>,
    a1_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES>,
    b_scales: &mut SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES>,
) -> ([f32; 4], [f32; 4]) {
    let mut acc0 = [0.0_f32; 4];
    let mut acc1 = [0.0_f32; 4];
    let mut k_base = 0;

    while k_base < params.input_dim {
        stage_b_tiles_aligned(
            weight_bytes,
            weight_scales,
            tile0,
            k_base,
            params,
            b_packs,
            b_scales,
        );
        stage_a_tiles_aligned(
            input_bytes,
            input_scales,
            tile0,
            k_base,
            params,
            a_packs,
            a_scales,
        );
        stage_a_tiles_aligned(
            input_bytes,
            input_scales,
            tile1,
            k_base,
            params,
            a1_packs,
            a1_scales,
        );
        thread::sync_threads();

        let mut k_atom = 0;
        while k_atom < NVFP4_PROJECTION_CTA_K_ATOMS {
            let b = load_b_fragments(b_packs, tile0, k_atom);
            let scale_b = load_b_scale4(b_scales, tile0, k_atom);
            mma_m16n8k64_scale4x_ue4m3(
                load_a_fragments(a_packs, tile0, k_atom),
                b,
                &mut acc0,
                load_a_scale4(a_scales, tile0, k_atom),
                scale_b,
            );
            mma_m16n8k64_scale4x_ue4m3(
                load_a_fragments(a1_packs, tile1, k_atom),
                b,
                &mut acc1,
                load_a_scale4(a1_scales, tile1, k_atom),
                scale_b,
            );
            k_atom += 1;
        }

        thread::sync_threads();
        k_base += NVFP4_PROJECTION_CTA_K;
    }

    (acc0, acc1)
}
