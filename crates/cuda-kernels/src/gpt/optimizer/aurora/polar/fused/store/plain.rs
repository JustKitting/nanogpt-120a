use crate::device_ptr::write_f32;
use crate::f16_tc_matmul::cta_tile::CtaTile;

use super::index::{col, row};

#[inline(always)]
pub(crate) fn store_plain(
    acc: [f32; 4],
    tile: CtaTile,
    warp_n: u32,
    out: *mut f32,
    rows: u32,
    cols: u32,
) {
    store_plain_one(acc[0], tile, warp_n, 0, out, rows, cols);
    store_plain_one(acc[1], tile, warp_n, 1, out, rows, cols);
    store_plain_one(acc[2], tile, warp_n, 2, out, rows, cols);
    store_plain_one(acc[3], tile, warp_n, 3, out, rows, cols);
}

#[inline(always)]
pub(crate) fn store_plain_transposed(
    acc: [f32; 4],
    tile: CtaTile,
    warp_n: u32,
    out: *mut f32,
    dim: u32,
) {
    store_plain_transposed_one(acc[0], tile, warp_n, 0, out, dim);
    store_plain_transposed_one(acc[1], tile, warp_n, 1, out, dim);
    store_plain_transposed_one(acc[2], tile, warp_n, 2, out, dim);
    store_plain_transposed_one(acc[3], tile, warp_n, 3, out, dim);
}

#[inline(always)]
fn store_plain_one(
    acc: f32,
    tile: CtaTile,
    warp_n: u32,
    acc_index: usize,
    out: *mut f32,
    rows: u32,
    cols: u32,
) {
    let row = row(tile, acc_index);
    let col = col(tile, warp_n, acc_index);
    if row < rows && col < cols {
        write_f32(out, row * cols + col, acc);
    }
}

#[inline(always)]
fn store_plain_transposed_one(
    acc: f32,
    tile: CtaTile,
    warp_n: u32,
    acc_index: usize,
    out: *mut f32,
    dim: u32,
) {
    let row = row(tile, acc_index);
    let col = col(tile, warp_n, acc_index);
    if row < dim && col < dim {
        write_f32(out, col * dim + row, acc);
    }
}
