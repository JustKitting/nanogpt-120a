use cuda_device::{SharedArray, thread};

use super::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, CTA_THREADS, CtaTile};

pub(super) fn stage_tiles(
    a: &[u16],
    b_t: &[u16],
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    tile: CtaTile,
    m: u32,
    n: u32,
    k: u32,
    k_base: u32,
) {
    let thread_id = thread::threadIdx_x();
    let mut offset = thread_id;
    while offset < CTA_A_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(offset, tile.row_base, k_base);
        a_tile[offset as usize] = if global_row < m && global_col < k {
            a[((tile.batch * m + global_row) * k + global_col) as usize]
        } else {
            0
        };
        offset += CTA_THREADS;
    }

    let mut offset = thread_id;
    while offset < CTA_B_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(offset, tile.col_base, k_base);
        b_tile[offset as usize] = if global_row < n && global_col < k {
            b_t[((tile.batch * n + global_row) * k + global_col) as usize]
        } else {
            0
        };
        offset += CTA_THREADS;
    }
}

pub(super) fn stage_tiles_aligned(
    a: &[u16],
    b_t: &[u16],
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    tile: CtaTile,
    m: u32,
    n: u32,
    k: u32,
    k_base: u32,
) {
    let thread_id = thread::threadIdx_x();
    let mut offset = thread_id;
    while offset < CTA_A_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(offset, tile.row_base, k_base);
        a_tile[offset as usize] = a[((tile.batch * m + global_row) * k + global_col) as usize];
        offset += CTA_THREADS;
    }

    let mut offset = thread_id;
    while offset < CTA_B_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(offset, tile.col_base, k_base);
        b_tile[offset as usize] = b_t[((tile.batch * n + global_row) * k + global_col) as usize];
        offset += CTA_THREADS;
    }
}

#[inline(always)]
pub(crate) fn stage_coords(offset: u32, row_base: u32, k_base: u32) -> (u32, u32) {
    let row = offset / CTA_K;
    let col = offset - row * CTA_K;
    (row_base + row, k_base + col)
}

#[inline(always)]
pub(crate) fn load_a_fragments(a_tile: &SharedArray<u16, CTA_A_ELEMS>, tile: CtaTile) -> [u32; 4] {
    [
        load_a_fragment(a_tile, tile, 0),
        load_a_fragment(a_tile, tile, 1),
        load_a_fragment(a_tile, tile, 2),
        load_a_fragment(a_tile, tile, 3),
    ]
}

#[inline(always)]
pub(crate) fn load_b_fragments(
    b_tile: &SharedArray<u16, CTA_B_ELEMS>,
    tile: CtaTile,
    warp_n: u32,
) -> [u32; 2] {
    [
        load_b_fragment(b_tile, tile, warp_n, 0),
        load_b_fragment(b_tile, tile, warp_n, 1),
    ]
}

#[inline(always)]
fn load_a_fragment(a_tile: &SharedArray<u16, CTA_A_ELEMS>, tile: CtaTile, register: u32) -> u32 {
    let row = tile.warp_m * 16 + tile.group + if register & 1 == 0 { 0 } else { 8 };
    let col = tile.thread_in_group * 2 + if register < 2 { 0 } else { 8 };
    load_packed2(a_tile, row * CTA_K + col)
}

#[inline(always)]
fn load_b_fragment(
    b_tile: &SharedArray<u16, CTA_B_ELEMS>,
    tile: CtaTile,
    warp_n: u32,
    register: u32,
) -> u32 {
    let row = warp_n * 8 + tile.group;
    let col = tile.thread_in_group * 2 + if register == 0 { 0 } else { 8 };
    load_packed2(b_tile, row * CTA_K + col)
}

#[inline(always)]
fn load_packed2<const N: usize>(tile: &SharedArray<u16, N>, offset: u32) -> u32 {
    (tile[offset as usize] as u32) | ((tile[offset as usize + 1] as u32) << 16)
}
