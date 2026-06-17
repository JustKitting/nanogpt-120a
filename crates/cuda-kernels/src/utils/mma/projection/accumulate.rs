use crate::mma::mma_m16n8k64_scale4x_ue4m3;

use super::args::{Nvfp4ProjectionParams, Nvfp4ProjectionTile};
use super::load::{load_a_fragments, load_a_scale4, load_b_fragments, load_b_scale4};

const MMA_K: u32 = 64;

#[inline(always)]
pub fn nvfp4_projection_accumulate_tile(
    input_bytes: &[u8],
    input_scales: &[u8],
    weight_bytes: &[u8],
    weight_scales: &[u8],
    tile: Nvfp4ProjectionTile,
    params: &Nvfp4ProjectionParams,
) -> [f32; 4] {
    let mut acc = [0.0_f32; 4];
    let mut k_base = 0;

    while k_base < params.input_dim {
        let a = load_a_fragments(
            input_bytes,
            tile.tile_row,
            k_base,
            tile.group,
            tile.thread_in_group,
            params,
        );
        let b = load_b_fragments(
            weight_bytes,
            tile.tile_col,
            k_base,
            tile.group,
            tile.thread_in_group,
            params,
        );
        let scale_a = load_a_scale4(
            input_scales,
            tile.tile_row,
            k_base,
            tile.group,
            tile.thread_in_group,
            params,
        );
        let scale_b = load_b_scale4(weight_scales, tile.tile_col, k_base, tile.group, params);

        mma_m16n8k64_scale4x_ue4m3(a, b, &mut acc, scale_a, scale_b);
        k_base += MMA_K;
    }

    acc
}
