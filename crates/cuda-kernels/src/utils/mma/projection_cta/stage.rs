use cuda_device::thread;

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
use super::{ProjectionCtaRowPairTiles, ProjectionCtaSources, ProjectionCtaTiles};

macro_rules! stage_tiles_fn {
    ($name:ident, $load_a_pack:ident, $load_b_pack:ident, $load_a_scale:ident, $load_b_scale:ident) => {
        pub fn $name(
            sources: ProjectionCtaSources<'_>,
            tile: Nvfp4ProjectionCtaTile,
            k_base: u32,
            params: &Nvfp4ProjectionParams,
            tiles: &mut ProjectionCtaTiles<'_>,
        ) {
            let thread_id = thread::threadIdx_x();
            let mut offset = thread_id;
            while offset < NVFP4_PROJECTION_CTA_A_PACKS as u32 {
                tiles.a_packs[offset as usize] =
                    $load_a_pack(sources.input_bytes, tile, offset, k_base, params);
                offset += NVFP4_PROJECTION_CTA_THREADS;
            }

            let mut offset = thread_id;
            while offset < NVFP4_PROJECTION_CTA_B_PACKS as u32 {
                tiles.b_packs[offset as usize] =
                    $load_b_pack(sources.weight_bytes, tile, offset, k_base, params);
                offset += NVFP4_PROJECTION_CTA_THREADS;
            }

            let mut offset = thread_id;
            while offset < NVFP4_PROJECTION_CTA_A_SCALES as u32 {
                tiles.a_scales[offset as usize] =
                    $load_a_scale(sources.input_scales, tile, offset, k_base, params);
                offset += NVFP4_PROJECTION_CTA_THREADS;
            }

            let mut offset = thread_id;
            while offset < NVFP4_PROJECTION_CTA_B_SCALES as u32 {
                tiles.b_scales[offset as usize] =
                    $load_b_scale(sources.weight_scales, tile, offset, k_base, params);
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

pub fn stage_row_pair_tiles_aligned(
    sources: ProjectionCtaSources<'_>,
    tile0: Nvfp4ProjectionCtaTile,
    tile1: Nvfp4ProjectionCtaTile,
    k_base: u32,
    params: &Nvfp4ProjectionParams,
    tiles: &mut ProjectionCtaRowPairTiles<'_>,
) {
    let thread_id = thread::threadIdx_x();
    let mut offset = thread_id;
    while offset < NVFP4_PROJECTION_CTA_B_PACKS as u32 {
        tiles.b_packs[offset as usize] =
            load_b_pack_aligned(sources.weight_bytes, tile0, offset, k_base, params);
        if offset < NVFP4_PROJECTION_CTA_A_PACKS as u32 {
            tiles.a0_packs[offset as usize] =
                load_a_pack_aligned(sources.input_bytes, tile0, offset, k_base, params);
            tiles.a1_packs[offset as usize] = if tile1.row_base < params.token_count {
                load_a_pack_aligned(sources.input_bytes, tile1, offset, k_base, params)
            } else {
                0
            };
        }
        offset += NVFP4_PROJECTION_CTA_THREADS;
    }

    let mut offset = thread_id;
    while offset < NVFP4_PROJECTION_CTA_B_SCALES as u32 {
        tiles.b_scales[offset as usize] =
            load_b_scale_aligned(sources.weight_scales, tile0, offset, k_base, params);
        if offset < NVFP4_PROJECTION_CTA_A_SCALES as u32 {
            tiles.a0_scales[offset as usize] =
                load_a_scale_aligned(sources.input_scales, tile0, offset, k_base, params);
            tiles.a1_scales[offset as usize] = if tile1.row_base < params.token_count {
                load_a_scale_aligned(sources.input_scales, tile1, offset, k_base, params)
            } else {
                crate::mma::projection::load_bytes::E4M3_ONE_PACKED4
            };
        }
        offset += NVFP4_PROJECTION_CTA_THREADS;
    }
}
