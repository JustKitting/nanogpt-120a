use cuda_device::DisjointSlice;

use super::cta_tile::CtaTile;

macro_rules! store_four {
    ($store:ident, $acc:ident, $tile:ident, $warp_n:expr, $($arg:expr),*) => {{
        $store($acc[0], $tile, $warp_n, 0, $($arg),*);
        $store($acc[1], $tile, $warp_n, 1, $($arg),*);
        $store($acc[2], $tile, $warp_n, 2, $($arg),*);
        $store($acc[3], $tile, $warp_n, 3, $($arg),*);
    }};
}

#[inline(always)]
pub(super) fn store_add(
    acc: [f32; 4],
    tile: CtaTile,
    warp_n: u32,
    base: &[f32],
    out: &mut DisjointSlice<f32>,
    rows: u32,
    cols: u32,
    base_scale: f32,
    matmul_scale: f32,
) {
    let scales = StoreScales {
        base: base_scale,
        matmul: matmul_scale,
    };
    store_four!(
        store_add_one,
        acc,
        tile,
        warp_n,
        base,
        out,
        rows,
        cols,
        scales
    );
}

#[derive(Clone, Copy)]
struct StoreScales {
    base: f32,
    matmul: f32,
}

#[inline(always)]
fn store_add_one(
    acc: f32,
    tile: CtaTile,
    warp_n: u32,
    acc_index: usize,
    base: &[f32],
    out: &mut DisjointSlice<f32>,
    rows: u32,
    cols: u32,
    scales: StoreScales,
) {
    let row = row(tile, acc_index);
    let col = col(tile, warp_n, acc_index);
    if row < rows && col < cols {
        let offset = ((tile.batch * rows + row) * cols + col) as usize;
        unsafe {
            *out.get_unchecked_mut(offset) = scales.base * base[offset] + scales.matmul * acc;
        }
    }
}

#[inline(always)]
fn row(tile: CtaTile, acc_index: usize) -> u32 {
    tile.row_base + tile.warp_m * 16 + tile.group + if acc_index < 2 { 0 } else { 8 }
}

#[inline(always)]
fn col(tile: CtaTile, warp_n: u32, acc_index: usize) -> u32 {
    tile.col_base + warp_n * 8 + tile.thread_in_group * 2 + (acc_index as u32 & 1)
}
