use cuda_device::thread;

use super::convert::cvt_rn_f16_f32;
use super::cta_stage::stage_coords;
use super::cta_tile::{CTA_A_ELEMS, CTA_THREADS, CtaTile};

macro_rules! stage_tiles_a_transposed_fn {
    ($name:ident, $rhs:ident: $rhs_ty:ty, $stage_rhs:path) => {
        pub(super) fn $name(
            a: &[f32],
            $rhs: &[$rhs_ty],
            a_tile: &mut super::CtaATile,
            b_tile: &mut super::CtaBTile,
            tile: CtaTile,
            m: u32,
            n: u32,
            k: u32,
            k_base: u32,
        ) {
            stage_a_transposed(a, a_tile, tile, m, k, k_base);
            $stage_rhs($rhs, b_tile, tile, n, k, k_base);
        }
    };
}

stage_tiles_a_transposed_fn!(stage_tiles_f32_a_transposed_rhs, rhs: f32, stage_rhs);
stage_tiles_a_transposed_fn!(
    stage_tiles_f32_a_transposed_half_rhs,
    rhs: u16,
    stage_half_rhs
);

fn stage_a_transposed(
    a: &[f32],
    a_tile: &mut super::CtaATile,
    tile: CtaTile,
    m: u32,
    k: u32,
    k_base: u32,
) {
    let mut offset = thread::threadIdx_x();
    while offset < CTA_A_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(offset, tile.row_base, k_base);
        a_tile[offset as usize] = if global_row < m && global_col < k {
            let index = ((tile.batch * k + global_col) * m + global_row) as usize;
            cvt_rn_f16_f32(a[index])
        } else {
            0
        };
        offset += CTA_THREADS;
    }
}

cta_stage_transposed_rhs_fn!(stage_rhs, f32, |rhs, index| cvt_rn_f16_f32(rhs[index]));
cta_stage_transposed_rhs_fn!(stage_half_rhs, u16, |rhs, index| rhs[index]);
