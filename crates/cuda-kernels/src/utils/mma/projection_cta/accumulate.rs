use cuda_device::thread;

use crate::mma::mma_m16n8k64_scale4x_ue4m3;
use crate::mma::projection::Nvfp4ProjectionParams;

use super::load::{load_a_fragments, load_a_scale4, load_b_fragments, load_b_scale4};
use super::stage::{stage_row_pair_tiles_aligned, stage_tiles, stage_tiles_aligned};
use super::tile::{NVFP4_PROJECTION_CTA_K, NVFP4_PROJECTION_CTA_K_ATOMS, Nvfp4ProjectionCtaTile};
use super::{ProjectionCtaRowPairTiles, ProjectionCtaSources, ProjectionCtaTiles};

macro_rules! projection_accumulator_fn {
    ($name:ident, $stage_tiles:ident) => {
        pub fn $name(
            sources: ProjectionCtaSources<'_>,
            tile: Nvfp4ProjectionCtaTile,
            params: &Nvfp4ProjectionParams,
            tiles: &mut ProjectionCtaTiles<'_>,
        ) -> [f32; 4] {
            let mut acc = [0.0_f32; 4];
            let mut k_base = 0;

            while k_base < params.input_dim {
                $stage_tiles(sources, tile, k_base, params, tiles);
                thread::sync_threads();
                let mut k_atom = 0;
                while k_atom < NVFP4_PROJECTION_CTA_K_ATOMS {
                    mma_m16n8k64_scale4x_ue4m3(
                        load_a_fragments(tiles.a_packs, tile, k_atom),
                        load_b_fragments(tiles.b_packs, tile, k_atom),
                        &mut acc,
                        load_a_scale4(tiles.a_scales, tile, k_atom),
                        load_b_scale4(tiles.b_scales, tile, k_atom),
                    );
                    k_atom += 1;
                }
                thread::sync_threads();
                k_base += NVFP4_PROJECTION_CTA_K;
            }

            acc
        }
    };
}

projection_accumulator_fn!(projection_accumulator, stage_tiles);
projection_accumulator_fn!(projection_accumulator_aligned, stage_tiles_aligned);

pub fn projection_accumulator_aligned_row_pair(
    sources: ProjectionCtaSources<'_>,
    tile0: Nvfp4ProjectionCtaTile,
    tile1: Nvfp4ProjectionCtaTile,
    params: &Nvfp4ProjectionParams,
    tiles: &mut ProjectionCtaRowPairTiles<'_>,
) -> ([f32; 4], [f32; 4]) {
    let mut acc0 = [0.0_f32; 4];
    let mut acc1 = [0.0_f32; 4];
    let mut k_base = 0;

    while k_base < params.input_dim {
        stage_row_pair_tiles_aligned(sources, tile0, tile1, k_base, params, tiles);
        thread::sync_threads();

        let mut k_atom = 0;
        while k_atom < NVFP4_PROJECTION_CTA_K_ATOMS {
            let b = load_b_fragments(tiles.b_packs, tile0, k_atom);
            let scale_b = load_b_scale4(tiles.b_scales, tile0, k_atom);
            mma_m16n8k64_scale4x_ue4m3(
                load_a_fragments(tiles.a0_packs, tile0, k_atom),
                b,
                &mut acc0,
                load_a_scale4(tiles.a0_scales, tile0, k_atom),
                scale_b,
            );
            mma_m16n8k64_scale4x_ue4m3(
                load_a_fragments(tiles.a1_packs, tile1, k_atom),
                b,
                &mut acc1,
                load_a_scale4(tiles.a1_scales, tile1, k_atom),
                scale_b,
            );
            k_atom += 1;
        }

        thread::sync_threads();
        k_base += NVFP4_PROJECTION_CTA_K;
    }

    (acc0, acc1)
}
