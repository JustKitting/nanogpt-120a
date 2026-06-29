use cuda_device::{SharedArray, thread};

use crate::mma::projection::Nvfp4ProjectionParams;
use crate::mma::projection::load_bytes::{
    E4M3_ONE_PACKED4, load_packed8, load_packed8_aligned, load_scale4, load_scale4_aligned,
};

use super::tile::{
    NVFP4_PROJECTION_CTA_A_PACKS, NVFP4_PROJECTION_CTA_A_SCALES, NVFP4_PROJECTION_CTA_B_PACKS,
    NVFP4_PROJECTION_CTA_B_SCALES, NVFP4_PROJECTION_CTA_M, NVFP4_PROJECTION_CTA_N,
    NVFP4_PROJECTION_CTA_PACKS_PER_ROW, NVFP4_PROJECTION_CTA_THREADS, Nvfp4ProjectionCtaTile,
};

const MMA_K: u32 = 64;

pub fn stage_tiles(
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
        a_packs[offset as usize] = load_a_pack(input_bytes, tile, offset, k_base, params);
        offset += NVFP4_PROJECTION_CTA_THREADS;
    }

    let mut offset = thread_id;
    while offset < NVFP4_PROJECTION_CTA_B_PACKS as u32 {
        b_packs[offset as usize] = load_b_pack(weight_bytes, tile, offset, k_base, params);
        offset += NVFP4_PROJECTION_CTA_THREADS;
    }

    let mut offset = thread_id;
    while offset < NVFP4_PROJECTION_CTA_A_SCALES as u32 {
        a_scales[offset as usize] = load_a_scale(input_scales, tile, offset, k_base, params);
        offset += NVFP4_PROJECTION_CTA_THREADS;
    }

    let mut offset = thread_id;
    while offset < NVFP4_PROJECTION_CTA_B_SCALES as u32 {
        b_scales[offset as usize] = load_b_scale(weight_scales, tile, offset, k_base, params);
        offset += NVFP4_PROJECTION_CTA_THREADS;
    }
}

pub fn stage_tiles_aligned(
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
        a_packs[offset as usize] = load_a_pack_aligned(input_bytes, tile, offset, k_base, params);
        offset += NVFP4_PROJECTION_CTA_THREADS;
    }

    let mut offset = thread_id;
    while offset < NVFP4_PROJECTION_CTA_B_PACKS as u32 {
        b_packs[offset as usize] = load_b_pack_aligned(weight_bytes, tile, offset, k_base, params);
        offset += NVFP4_PROJECTION_CTA_THREADS;
    }

    let mut offset = thread_id;
    while offset < NVFP4_PROJECTION_CTA_A_SCALES as u32 {
        a_scales[offset as usize] =
            load_a_scale_aligned(input_scales, tile, offset, k_base, params);
        offset += NVFP4_PROJECTION_CTA_THREADS;
    }

    let mut offset = thread_id;
    while offset < NVFP4_PROJECTION_CTA_B_SCALES as u32 {
        b_scales[offset as usize] =
            load_b_scale_aligned(weight_scales, tile, offset, k_base, params);
        offset += NVFP4_PROJECTION_CTA_THREADS;
    }
}

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

#[inline(always)]
fn load_a_pack(
    bytes: &[u8],
    tile: Nvfp4ProjectionCtaTile,
    offset: u32,
    k_base: u32,
    params: &Nvfp4ProjectionParams,
) -> u32 {
    let (global_row, global_col) = pack_coords(offset, tile.row_base, k_base);
    if global_row < params.token_count && global_col + 7 < params.input_dim {
        load_packed8(bytes, (global_row * params.input_dim + global_col) as usize)
    } else {
        0
    }
}

#[inline(always)]
fn load_a_pack_aligned(
    bytes: &[u8],
    tile: Nvfp4ProjectionCtaTile,
    offset: u32,
    k_base: u32,
    params: &Nvfp4ProjectionParams,
) -> u32 {
    let (global_row, global_col) = pack_coords(offset, tile.row_base, k_base);
    load_packed8_aligned(bytes, (global_row * params.input_dim + global_col) as usize)
}

#[inline(always)]
fn load_b_pack(
    bytes: &[u8],
    tile: Nvfp4ProjectionCtaTile,
    offset: u32,
    k_base: u32,
    params: &Nvfp4ProjectionParams,
) -> u32 {
    let (global_col, global_k) = pack_coords(offset, tile.col_base, k_base);
    if global_col < params.output_dim && global_k + 7 < params.input_dim {
        load_packed8(bytes, (global_col * params.input_dim + global_k) as usize)
    } else {
        0
    }
}

#[inline(always)]
fn load_b_pack_aligned(
    bytes: &[u8],
    tile: Nvfp4ProjectionCtaTile,
    offset: u32,
    k_base: u32,
    params: &Nvfp4ProjectionParams,
) -> u32 {
    let (global_col, global_k) = pack_coords(offset, tile.col_base, k_base);
    load_packed8_aligned(bytes, (global_col * params.input_dim + global_k) as usize)
}

#[inline(always)]
fn load_a_scale(
    scales: &[u8],
    tile: Nvfp4ProjectionCtaTile,
    offset: u32,
    k_base: u32,
    params: &Nvfp4ProjectionParams,
) -> u32 {
    let (global_row, scale_k_base) =
        scale_coords(offset, tile.row_base, NVFP4_PROJECTION_CTA_M, k_base);
    if global_row < params.token_count && scale_k_base < params.input_dim {
        load_scale4(
            scales,
            ((global_row * params.input_dim + scale_k_base) / 16) as usize,
        )
    } else {
        E4M3_ONE_PACKED4
    }
}

#[inline(always)]
fn load_a_scale_aligned(
    scales: &[u8],
    tile: Nvfp4ProjectionCtaTile,
    offset: u32,
    k_base: u32,
    params: &Nvfp4ProjectionParams,
) -> u32 {
    let (global_row, scale_k_base) =
        scale_coords(offset, tile.row_base, NVFP4_PROJECTION_CTA_M, k_base);
    load_scale4_aligned(
        scales,
        ((global_row * params.input_dim + scale_k_base) / 16) as usize,
    )
}

#[inline(always)]
fn load_b_scale(
    scales: &[u8],
    tile: Nvfp4ProjectionCtaTile,
    offset: u32,
    k_base: u32,
    params: &Nvfp4ProjectionParams,
) -> u32 {
    let (global_col, scale_k_base) =
        scale_coords(offset, tile.col_base, NVFP4_PROJECTION_CTA_N, k_base);
    if global_col < params.output_dim && scale_k_base < params.input_dim {
        let scale_base = global_col * (params.input_dim / 16) + scale_k_base / 16;
        load_scale4(scales, scale_base as usize)
    } else {
        E4M3_ONE_PACKED4
    }
}

#[inline(always)]
fn load_b_scale_aligned(
    scales: &[u8],
    tile: Nvfp4ProjectionCtaTile,
    offset: u32,
    k_base: u32,
    params: &Nvfp4ProjectionParams,
) -> u32 {
    let (global_col, scale_k_base) =
        scale_coords(offset, tile.col_base, NVFP4_PROJECTION_CTA_N, k_base);
    let scale_base = global_col * (params.input_dim / 16) + scale_k_base / 16;
    load_scale4_aligned(scales, scale_base as usize)
}

#[inline(always)]
fn pack_coords(offset: u32, row_base: u32, k_base: u32) -> (u32, u32) {
    let row = offset / NVFP4_PROJECTION_CTA_PACKS_PER_ROW;
    let pack = offset - row * NVFP4_PROJECTION_CTA_PACKS_PER_ROW;
    (row_base + row, k_base + pack * 8)
}

#[inline(always)]
fn scale_coords(offset: u32, row_base: u32, rows_per_atom: u32, k_base: u32) -> (u32, u32) {
    let k_atom = offset / rows_per_atom;
    let row = offset - k_atom * rows_per_atom;
    (row_base + row, k_base + k_atom * MMA_K)
}
