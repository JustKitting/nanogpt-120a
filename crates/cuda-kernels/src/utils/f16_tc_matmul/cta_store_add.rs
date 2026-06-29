use cuda_device::DisjointSlice;

use super::cta_tile::CtaTile;

struct StoreAddArgs<'a> {
    tile: CtaTile,
    warp_n: u32,
    base: &'a [f32],
    rows: u32,
    cols: u32,
    base_scale: f32,
    matmul_scale: f32,
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
    let args = StoreAddArgs {
        tile,
        warp_n,
        base,
        rows,
        cols,
        base_scale,
        matmul_scale,
    };
    store_add_one(acc[0], 0, out, &args);
    store_add_one(acc[1], 1, out, &args);
    store_add_one(acc[2], 2, out, &args);
    store_add_one(acc[3], 3, out, &args);
}

#[inline(always)]
fn store_add_one(
    acc: f32,
    acc_index: usize,
    out: &mut DisjointSlice<f32>,
    args: &StoreAddArgs<'_>,
) {
    let row = args.tile.row_base
        + args.tile.warp_m * 16
        + args.tile.group
        + if acc_index < 2 { 0 } else { 8 };
    let col = args.tile.col_base
        + args.warp_n * 8
        + args.tile.thread_in_group * 2
        + (acc_index as u32 & 1);
    if row < args.rows && col < args.cols {
        let offset = ((args.tile.batch * args.rows + row) * args.cols + col) as usize;
        unsafe {
            *out.get_unchecked_mut(offset) =
                args.base_scale * args.base[offset] + args.matmul_scale * acc;
        }
    }
}
