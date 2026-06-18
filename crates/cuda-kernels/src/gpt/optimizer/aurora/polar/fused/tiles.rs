use cuda_device::{SharedArray, thread};

use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_M, CTA_N, CtaTile};

use super::super::super::super::work_grid::WorkGrid;
use super::coefficients::Coefficients;
use super::store::{store_next, store_plain};

mod compute;
use compute::compute_tile;
mod symmetric;
pub(super) use symmetric::run_symmetric_tiles;

#[allow(clippy::too_many_arguments)]
pub(super) fn run_plain_tiles(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    work: WorkGrid,
    m: u32,
    n: u32,
    k: u32,
    rhs_transposed: bool,
) {
    let n_tiles = n.div_ceil(CTA_N);
    let tile_count = m.div_ceil(CTA_M) * n_tiles;
    let mut tile_index = work.block();

    while tile_index < tile_count {
        let tile_row = tile_index / n_tiles;
        let tile_col = tile_index - tile_row * n_tiles;
        let tile = CtaTile::from_tile(thread::threadIdx_x(), tile_row, tile_col, 0);
        let (acc0, acc1, acc2, acc3) =
            compute_tile(a, b, a_tile, b_tile, tile, m, n, k, rhs_transposed);
        store_plain(acc0, tile, tile.warp_n0, out, m, n);
        store_plain(acc1, tile, tile.warp_n0 + 1, out, m, n);
        store_plain(acc2, tile, tile.warp_n0 + 2, out, m, n);
        store_plain(acc3, tile, tile.warp_n0 + 3, out, m, n);
        tile_index += work.blocks();
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_next_tiles(
    a: *const f32,
    rhs: *const f32,
    base0: *const f32,
    base1: *const f32,
    out: *mut f32,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    work: WorkGrid,
    rows: u32,
    cols: u32,
    coefficients: Coefficients,
) {
    let n_tiles = cols.div_ceil(CTA_N);
    let tile_count = rows.div_ceil(CTA_M) * n_tiles;
    let mut tile_index = work.block();

    while tile_index < tile_count {
        let tile_row = tile_index / n_tiles;
        let tile_col = tile_index - tile_row * n_tiles;
        let tile = CtaTile::from_tile(thread::threadIdx_x(), tile_row, tile_col, 0);
        let (acc0, acc1, acc2, acc3) =
            compute_tile(a, rhs, a_tile, b_tile, tile, rows, cols, rows, true);
        store_next(
            acc0,
            tile,
            tile.warp_n0,
            base0,
            base1,
            out,
            rows,
            cols,
            coefficients,
        );
        store_next(
            acc1,
            tile,
            tile.warp_n0 + 1,
            base0,
            base1,
            out,
            rows,
            cols,
            coefficients,
        );
        store_next(
            acc2,
            tile,
            tile.warp_n0 + 2,
            base0,
            base1,
            out,
            rows,
            cols,
            coefficients,
        );
        store_next(
            acc3,
            tile,
            tile.warp_n0 + 3,
            base0,
            base1,
            out,
            rows,
            cols,
            coefficients,
        );
        tile_index += work.blocks();
    }
}
