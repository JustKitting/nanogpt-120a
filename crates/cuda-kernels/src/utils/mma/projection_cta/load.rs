use cuda_device::SharedArray;

use super::tile::{
    NVFP4_PROJECTION_CTA_A_PACKS, NVFP4_PROJECTION_CTA_A_SCALES, NVFP4_PROJECTION_CTA_B_PACKS,
    NVFP4_PROJECTION_CTA_B_SCALES, NVFP4_PROJECTION_CTA_M, NVFP4_PROJECTION_CTA_N,
    NVFP4_PROJECTION_CTA_PACKS_PER_ROW, Nvfp4ProjectionCtaTile,
};

const MMA_PACKS_PER_ROW: u32 = 8;

#[inline(always)]
pub fn load_a_fragments(
    a_packs: &SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS>,
    tile: Nvfp4ProjectionCtaTile,
    k_atom: u32,
) -> [u32; 4] {
    [
        load_a_fragment(a_packs, tile, k_atom, 0),
        load_a_fragment(a_packs, tile, k_atom, 1),
        load_a_fragment(a_packs, tile, k_atom, 2),
        load_a_fragment(a_packs, tile, k_atom, 3),
    ]
}

#[inline(always)]
pub fn load_b_fragments(
    b_packs: &SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS>,
    tile: Nvfp4ProjectionCtaTile,
    k_atom: u32,
) -> [u32; 2] {
    [
        load_b_fragment(b_packs, tile, k_atom, 0),
        load_b_fragment(b_packs, tile, k_atom, 1),
    ]
}

#[inline(always)]
pub fn load_a_scale4(
    a_scales: &SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES>,
    tile: Nvfp4ProjectionCtaTile,
    k_atom: u32,
) -> u32 {
    let row = tile.warp_m * 16 + tile.group + if tile.thread_in_group == 1 { 8 } else { 0 };
    a_scales[(k_atom * NVFP4_PROJECTION_CTA_M + row) as usize]
}

#[inline(always)]
pub fn load_b_scale4(
    b_scales: &SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES>,
    tile: Nvfp4ProjectionCtaTile,
    k_atom: u32,
) -> u32 {
    let col = tile.warp_n * 8 + tile.group;
    b_scales[(k_atom * NVFP4_PROJECTION_CTA_N + col) as usize]
}

#[inline(always)]
fn load_a_fragment(
    a_packs: &SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS>,
    tile: Nvfp4ProjectionCtaTile,
    k_atom: u32,
    register: u32,
) -> u32 {
    let row = tile.warp_m * 16 + tile.group + if register & 1 == 0 { 0 } else { 8 };
    let pack = k_atom * MMA_PACKS_PER_ROW + tile.thread_in_group + if register < 2 { 0 } else { 4 };
    a_packs[(row * NVFP4_PROJECTION_CTA_PACKS_PER_ROW + pack) as usize]
}

#[inline(always)]
fn load_b_fragment(
    b_packs: &SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS>,
    tile: Nvfp4ProjectionCtaTile,
    k_atom: u32,
    register: u32,
) -> u32 {
    let col = tile.warp_n * 8 + tile.group;
    let pack =
        k_atom * MMA_PACKS_PER_ROW + tile.thread_in_group + if register == 0 { 0 } else { 4 };
    b_packs[(col * NVFP4_PROJECTION_CTA_PACKS_PER_ROW + pack) as usize]
}
