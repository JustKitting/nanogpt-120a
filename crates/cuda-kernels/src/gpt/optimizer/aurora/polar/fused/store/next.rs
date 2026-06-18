use crate::device_ptr::{read_f32, write_f32};
use crate::f16_tc_matmul::cta_tile::CtaTile;
use crate::float_ptx::fma_f32;

use super::super::coefficients::Coefficients;
use super::index::{col, row};

#[allow(clippy::too_many_arguments)]
pub(crate) fn store_next(
    acc: [f32; 4],
    tile: CtaTile,
    warp_n: u32,
    base0: *const f32,
    base1: *const f32,
    out: *mut f32,
    rows: u32,
    cols: u32,
    coefficients: Coefficients,
) {
    store_next_one(
        acc[0],
        tile,
        warp_n,
        0,
        base0,
        base1,
        out,
        rows,
        cols,
        coefficients,
    );
    store_next_one(
        acc[1],
        tile,
        warp_n,
        1,
        base0,
        base1,
        out,
        rows,
        cols,
        coefficients,
    );
    store_next_one(
        acc[2],
        tile,
        warp_n,
        2,
        base0,
        base1,
        out,
        rows,
        cols,
        coefficients,
    );
    store_next_one(
        acc[3],
        tile,
        warp_n,
        3,
        base0,
        base1,
        out,
        rows,
        cols,
        coefficients,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn store_next_one(
    acc: f32,
    tile: CtaTile,
    warp_n: u32,
    acc_index: usize,
    base0: *const f32,
    base1: *const f32,
    out: *mut f32,
    rows: u32,
    cols: u32,
    coefficients: Coefficients,
) {
    let row = row(tile, acc_index);
    let col = col(tile, warp_n, acc_index);
    if row < rows && col < cols {
        let offset = row * cols + col;
        let base = fma_f32(
            coefficients.a,
            read_f32(base0, offset),
            coefficients.b * read_f32(base1, offset),
        );
        write_f32(out, offset, fma_f32(coefficients.c, acc, base));
    }
}
