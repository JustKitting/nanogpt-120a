use cuda_device::DisjointSlice;

use super::cta_tile::CtaTile;

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
    store_add_one(
        acc[0],
        tile,
        warp_n,
        0,
        base,
        out,
        rows,
        cols,
        base_scale,
        matmul_scale,
    );
    store_add_one(
        acc[1],
        tile,
        warp_n,
        1,
        base,
        out,
        rows,
        cols,
        base_scale,
        matmul_scale,
    );
    store_add_one(
        acc[2],
        tile,
        warp_n,
        2,
        base,
        out,
        rows,
        cols,
        base_scale,
        matmul_scale,
    );
    store_add_one(
        acc[3],
        tile,
        warp_n,
        3,
        base,
        out,
        rows,
        cols,
        base_scale,
        matmul_scale,
    );
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
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
    base_scale: f32,
    matmul_scale: f32,
) {
    let row = tile.row_base + tile.warp_m * 16 + tile.group + if acc_index < 2 { 0 } else { 8 };
    let col = tile.col_base + warp_n * 8 + tile.thread_in_group * 2 + (acc_index as u32 & 1);
    if row < rows && col < cols {
        let offset = ((tile.batch * rows + row) * cols + col) as usize;
        unsafe {
            *out.get_unchecked_mut(offset) = base_scale * base[offset] + matmul_scale * acc;
        }
    }
}
