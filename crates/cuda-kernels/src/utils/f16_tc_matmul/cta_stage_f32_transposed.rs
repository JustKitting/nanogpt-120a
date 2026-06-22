use cuda_device::{SharedArray, thread};

use super::convert::cvt_rn_f16_f32;
use super::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, CTA_THREADS, CtaTile};

pub(super) fn stage_tiles_f32_a_transposed_rhs(
    a: &[f32],
    rhs: &[f32],
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    tile: CtaTile,
    m: u32,
    n: u32,
    k: u32,
    k_base: u32,
) {
    stage_a_transposed(a, a_tile, tile, m, k, k_base);
    stage_rhs(rhs, b_tile, tile, n, k, k_base);
}

pub(super) fn stage_tiles_f32_a_transposed_half_rhs(
    a: &[f32],
    rhs: &[u16],
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    tile: CtaTile,
    m: u32,
    n: u32,
    k: u32,
    k_base: u32,
) {
    stage_a_transposed(a, a_tile, tile, m, k, k_base);
    stage_half_rhs(rhs, b_tile, tile, n, k, k_base);
}

fn stage_a_transposed(
    a: &[f32],
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    tile: CtaTile,
    m: u32,
    k: u32,
    k_base: u32,
) {
    let mut offset = thread::threadIdx_x();
    while offset < CTA_A_ELEMS as u32 {
        let row = offset / CTA_K;
        let col = offset - row * CTA_K;
        let global_row = tile.row_base + row;
        let global_col = k_base + col;
        a_tile[offset as usize] = if global_row < m && global_col < k {
            let index = ((tile.batch * k + global_col) * m + global_row) as usize;
            cvt_rn_f16_f32(a[index])
        } else {
            0
        };
        offset += CTA_THREADS;
    }
}

fn stage_rhs(
    rhs: &[f32],
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    tile: CtaTile,
    n: u32,
    k: u32,
    k_base: u32,
) {
    let mut offset = thread::threadIdx_x();
    while offset < CTA_B_ELEMS as u32 {
        let row = offset / CTA_K;
        let col = offset - row * CTA_K;
        let global_row = tile.col_base + row;
        let global_col = k_base + col;
        b_tile[offset as usize] = if global_row < n && global_col < k {
            let index = ((tile.batch * k + global_col) * n + global_row) as usize;
            cvt_rn_f16_f32(rhs[index])
        } else {
            0
        };
        offset += CTA_THREADS;
    }
}

fn stage_half_rhs(
    rhs: &[u16],
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    tile: CtaTile,
    n: u32,
    k: u32,
    k_base: u32,
) {
    let mut offset = thread::threadIdx_x();
    while offset < CTA_B_ELEMS as u32 {
        let row = offset / CTA_K;
        let col = offset - row * CTA_K;
        let global_row = tile.col_base + row;
        let global_col = k_base + col;
        b_tile[offset as usize] = if global_row < n && global_col < k {
            let index = ((tile.batch * k + global_col) * n + global_row) as usize;
            rhs[index]
        } else {
            0
        };
        offset += CTA_THREADS;
    }
}
