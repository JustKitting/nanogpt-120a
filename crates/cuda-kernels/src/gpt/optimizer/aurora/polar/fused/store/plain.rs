use crate::device_ptr::{read_f32, write_f32};
use crate::f16_tc_matmul::cta_tile::CtaTile;
use crate::float_ptx::fma_f32;

use super::super::coefficients::Coefficients;
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
pub(crate) fn store_symmetric_polynomial(
    acc: [f32; 4],
    tile: CtaTile,
    warp_n: u32,
    base: *const f32,
    out: *mut f32,
    dim: u32,
    coefficients: Coefficients,
) {
    store_symmetric_polynomial_one(acc[0], tile, warp_n, 0, base, out, dim, coefficients);
    store_symmetric_polynomial_one(acc[1], tile, warp_n, 1, base, out, dim, coefficients);
    store_symmetric_polynomial_one(acc[2], tile, warp_n, 2, base, out, dim, coefficients);
    store_symmetric_polynomial_one(acc[3], tile, warp_n, 3, base, out, dim, coefficients);
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

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn store_symmetric_polynomial_one(
    acc: f32,
    tile: CtaTile,
    warp_n: u32,
    acc_index: usize,
    base: *const f32,
    out: *mut f32,
    dim: u32,
    coefficients: Coefficients,
) {
    let row = row(tile, acc_index);
    let col = col(tile, warp_n, acc_index);
    if row < dim && col < dim {
        let offset = row * dim + col;
        write_f32(
            out,
            offset,
            polynomial_value(acc, read_f32(base, offset), row == col, coefficients),
        );
    }
    if row < dim && col < dim && row != col {
        let offset = col * dim + row;
        write_f32(
            out,
            offset,
            polynomial_value(acc, read_f32(base, offset), false, coefficients),
        );
    }
}

#[inline(always)]
fn polynomial_value(acc: f32, base: f32, diagonal: bool, coefficients: Coefficients) -> f32 {
    let identity = if diagonal { coefficients.a } else { 0.0 };
    fma_f32(coefficients.c, acc, fma_f32(coefficients.b, base, identity))
}
