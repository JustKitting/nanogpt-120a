use cuda_device::{SharedArray, thread};

use super::convert::cvt_rn_f16_f32;
use super::cta_stage::stage_coords;
use super::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_THREADS, CtaTile};

macro_rules! stage_tiles_f32_fn {
    ($name:ident, $lhs:ident: $lhs_ty:ty, $rhs:ident: $rhs_ty:ty, $stage_lhs:path, $stage_rhs:path) => {
        pub(super) fn $name(
            $lhs: &[$lhs_ty],
            $rhs: &[$rhs_ty],
            a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
            b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
            tile: CtaTile,
            m: u32,
            n: u32,
            k: u32,
            k_base: u32,
        ) {
            $stage_lhs($lhs, a_tile, tile, m, k, k_base);
            $stage_rhs($rhs, b_tile, tile, n, k, k_base);
        }
    };
}

stage_tiles_f32_fn!(stage_tiles_f32_b_t, a: f32, b_t: f32, stage_a, stage_b_t);
stage_tiles_f32_fn!(
    stage_tiles_f32_b_t_aligned,
    a: f32,
    b_t: f32,
    stage_a_aligned,
    stage_b_t_aligned
);
stage_tiles_f32_fn!(
    stage_tiles_f32_rhs_transposed,
    a: f32,
    rhs: f32,
    stage_a,
    stage_rhs_transposed
);
stage_tiles_f32_fn!(
    stage_tiles_f32_half_rhs,
    a: f32,
    rhs: u16,
    stage_a,
    stage_half_rhs_transposed
);

fn stage_a(
    a: &[f32],
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    tile: CtaTile,
    m: u32,
    k: u32,
    k_base: u32,
) {
    let mut offset = thread::threadIdx_x();
    while offset < CTA_A_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(offset, tile.row_base, k_base);
        a_tile[offset as usize] = if global_row < m && global_col < k {
            cvt_rn_f16_f32(a[((tile.batch * m + global_row) * k + global_col) as usize])
        } else {
            0
        };
        offset += CTA_THREADS;
    }
}

fn stage_a_aligned(
    a: &[f32],
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    tile: CtaTile,
    m: u32,
    k: u32,
    k_base: u32,
) {
    let mut offset = thread::threadIdx_x();
    while offset < CTA_A_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(offset, tile.row_base, k_base);
        a_tile[offset as usize] =
            cvt_rn_f16_f32(a[((tile.batch * m + global_row) * k + global_col) as usize]);
        offset += CTA_THREADS;
    }
}

fn stage_b_t(
    b_t: &[f32],
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    tile: CtaTile,
    n: u32,
    k: u32,
    k_base: u32,
) {
    let mut offset = thread::threadIdx_x();
    while offset < CTA_B_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(offset, tile.col_base, k_base);
        b_tile[offset as usize] = if global_row < n && global_col < k {
            cvt_rn_f16_f32(b_t[((tile.batch * n + global_row) * k + global_col) as usize])
        } else {
            0
        };
        offset += CTA_THREADS;
    }
}

fn stage_b_t_aligned(
    b_t: &[f32],
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    tile: CtaTile,
    n: u32,
    k: u32,
    k_base: u32,
) {
    let mut offset = thread::threadIdx_x();
    while offset < CTA_B_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(offset, tile.col_base, k_base);
        b_tile[offset as usize] =
            cvt_rn_f16_f32(b_t[((tile.batch * n + global_row) * k + global_col) as usize]);
        offset += CTA_THREADS;
    }
}

macro_rules! stage_rhs_transposed_fn {
    ($name:ident, $rhs_ty:ty, |$rhs:ident, $index:ident| $value:expr) => {
        fn $name(
            $rhs: &[$rhs_ty],
            b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
            tile: CtaTile,
            n: u32,
            k: u32,
            k_base: u32,
        ) {
            let mut offset = thread::threadIdx_x();
            while offset < CTA_B_ELEMS as u32 {
                let (global_row, global_col) = stage_coords(offset, tile.col_base, k_base);
                b_tile[offset as usize] = if global_row < n && global_col < k {
                    let $index = ((tile.batch * k + global_col) * n + global_row) as usize;
                    $value
                } else {
                    0
                };
                offset += CTA_THREADS;
            }
        }
    };
}

stage_rhs_transposed_fn!(stage_rhs_transposed, f32, |rhs, index| cvt_rn_f16_f32(
    rhs[index]
));
stage_rhs_transposed_fn!(stage_half_rhs_transposed, u16, |rhs, index| rhs[index]);
