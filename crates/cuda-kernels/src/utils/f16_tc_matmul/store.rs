use cuda_device::DisjointSlice;

use super::tile::Tile;

#[inline(always)]
pub(super) fn store(
    acc: [f32; 4],
    batch: u32,
    tile: Tile,
    out: &mut DisjointSlice<f32>,
    rows: u32,
    cols: u32,
) {
    store_one(acc[0], batch, tile, out, rows, cols, 0);
    store_one(acc[1], batch, tile, out, rows, cols, 1);
    store_one(acc[2], batch, tile, out, rows, cols, 2);
    store_one(acc[3], batch, tile, out, rows, cols, 3);
}

#[inline(always)]
fn store_one(
    acc: f32,
    batch: u32,
    tile: Tile,
    out: &mut DisjointSlice<f32>,
    rows: u32,
    cols: u32,
    acc_index: usize,
) {
    let row = tile.row + tile.group + if acc_index < 2 { 0 } else { 8 };
    let col = tile.col + tile.thread_in_group * 2 + (acc_index as u32 & 1);
    if row < rows && col < cols {
        let offset = ((batch * rows + row) * cols + col) as usize;
        unsafe {
            *out.get_unchecked_mut(offset) = acc;
        }
    }
}
