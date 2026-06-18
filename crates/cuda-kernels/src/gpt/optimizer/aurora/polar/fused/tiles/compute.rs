use cuda_device::{SharedArray, thread};

use crate::f16_tc_matmul::cta_stage::{load_a_fragments, load_b_fragments};
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, CtaTile};
use crate::mma::mma_m16n8k16_f16_f16_f32;

use super::super::stage::stage_tiles;

#[allow(clippy::too_many_arguments)]
pub(super) fn compute_tile(
    a: *const f32,
    b: *const f32,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    tile: CtaTile,
    m: u32,
    n: u32,
    k: u32,
    rhs_transposed: bool,
) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
    let mut acc0 = [0.0_f32; 4];
    let mut acc1 = [0.0_f32; 4];
    let mut acc2 = [0.0_f32; 4];
    let mut acc3 = [0.0_f32; 4];
    let mut k_base = 0;
    while k_base < k {
        stage_tiles(a, b, a_tile, b_tile, tile, m, n, k, k_base, rhs_transposed);
        thread::sync_threads();
        let a_fragments = load_a_fragments(a_tile, tile);
        mma_m16n8k16_f16_f16_f32(
            a_fragments,
            load_b_fragments(b_tile, tile, tile.warp_n0),
            &mut acc0,
        );
        mma_m16n8k16_f16_f16_f32(
            a_fragments,
            load_b_fragments(b_tile, tile, tile.warp_n0 + 1),
            &mut acc1,
        );
        mma_m16n8k16_f16_f16_f32(
            a_fragments,
            load_b_fragments(b_tile, tile, tile.warp_n0 + 2),
            &mut acc2,
        );
        mma_m16n8k16_f16_f16_f32(
            a_fragments,
            load_b_fragments(b_tile, tile, tile.warp_n0 + 3),
            &mut acc3,
        );
        thread::sync_threads();
        k_base += CTA_K;
    }
    (acc0, acc1, acc2, acc3)
}
