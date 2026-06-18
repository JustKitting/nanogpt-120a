use cuda_device::{SharedArray, thread};

use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_M, CtaTile};

use super::super::super::super::super::work_grid::WorkGrid;
use super::super::store::{store_plain, store_plain_transposed};
use super::compute_tile;

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_symmetric_tiles(
    source: *const f32,
    out: *mut f32,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    work: WorkGrid,
    dim: u32,
    k: u32,
) {
    let tile_dim = dim.div_ceil(CTA_M);
    let tile_count = tile_dim * tile_dim;
    let mut tile_index = work.block();

    while tile_index < tile_count {
        let tile_row = tile_index / tile_dim;
        let tile_col = tile_index - tile_row * tile_dim;
        if tile_col >= tile_row {
            run_tile(source, out, a_tile, b_tile, dim, k, tile_row, tile_col);
        }
        tile_index += work.blocks();
    }
}

#[allow(clippy::too_many_arguments)]
fn run_tile(
    source: *const f32,
    out: *mut f32,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    dim: u32,
    k: u32,
    tile_row: u32,
    tile_col: u32,
) {
    let tile = CtaTile::from_tile(thread::threadIdx_x(), tile_row, tile_col, 0);
    let (acc0, acc1, acc2, acc3) =
        compute_tile(source, source, a_tile, b_tile, tile, dim, dim, k, false);
    store_plain(acc0, tile, tile.warp_n0, out, dim, dim);
    store_plain(acc1, tile, tile.warp_n0 + 1, out, dim, dim);
    store_plain(acc2, tile, tile.warp_n0 + 2, out, dim, dim);
    store_plain(acc3, tile, tile.warp_n0 + 3, out, dim, dim);
    if tile_col != tile_row {
        store_plain_transposed(acc0, tile, tile.warp_n0, out, dim);
        store_plain_transposed(acc1, tile, tile.warp_n0 + 1, out, dim);
        store_plain_transposed(acc2, tile, tile.warp_n0 + 2, out, dim);
        store_plain_transposed(acc3, tile, tile.warp_n0 + 3, out, dim);
    }
}
