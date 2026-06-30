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

macro_rules! stage_row_major_f32_fn {
    ($name:ident, $tile_elems:ident, $row_base:ident, $check_bounds:expr) => {
        fn $name(
            src: &[f32],
            dst: &mut SharedArray<u16, $tile_elems>,
            tile: CtaTile,
            rows: u32,
            cols: u32,
            k_base: u32,
        ) {
            stage_row_major_f32::<$check_bounds, $tile_elems>(
                src,
                dst,
                tile,
                tile.$row_base,
                rows,
                cols,
                k_base,
            );
        }
    };
}

stage_row_major_f32_fn!(stage_a, CTA_A_ELEMS, row_base, true);
stage_row_major_f32_fn!(stage_a_aligned, CTA_A_ELEMS, row_base, false);
stage_row_major_f32_fn!(stage_b_t, CTA_B_ELEMS, col_base, true);
stage_row_major_f32_fn!(stage_b_t_aligned, CTA_B_ELEMS, col_base, false);

fn stage_row_major_f32<const CHECK_BOUNDS: bool, const TILE_ELEMS: usize>(
    src: &[f32],
    dst: &mut SharedArray<u16, TILE_ELEMS>,
    tile: CtaTile,
    row_base: u32,
    rows: u32,
    cols: u32,
    k_base: u32,
) {
    let mut offset = thread::threadIdx_x();
    while offset < TILE_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(offset, row_base, k_base);
        dst[offset as usize] = if !CHECK_BOUNDS || (global_row < rows && global_col < cols) {
            cvt_rn_f16_f32(src[((tile.batch * rows + global_row) * cols + global_col) as usize])
        } else {
            0
        };
        offset += CTA_THREADS;
    }
}

cta_stage_transposed_rhs_fn!(stage_rhs_transposed, f32, |rhs, index| cvt_rn_f16_f32(
    rhs[index]
));
cta_stage_transposed_rhs_fn!(stage_half_rhs_transposed, u16, |rhs, index| rhs[index]);
