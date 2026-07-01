use cuda_device::DisjointSlice;

use super::cta_tile::{CtaMatmulDims, CtaTile};

pub(crate) struct StoreAddArgs<'a> {
    tile: CtaTile,
    warp_n: u32,
    base: &'a [f32],
    dims: CtaMatmulDims,
    base_scale: f32,
    matmul_scale: f32,
}

impl<'a> StoreAddArgs<'a> {
    #[inline(always)]
    pub(crate) fn new(tile: CtaTile, warp_n: u32, base: &'a [f32], dims: CtaMatmulDims, base_scale: f32, matmul_scale: f32) -> Self {
        Self { tile, warp_n, base, dims, base_scale, matmul_scale }
    }
}

#[inline(always)]
pub(super) fn store_add(
    acc: [f32; 4],
    out: &mut DisjointSlice<f32>,
    args: StoreAddArgs<'_>,
) {
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
    let (row, col) = args.tile.accumulator_coords(args.warp_n, acc_index);
    if row < args.dims.m && col < args.dims.n {
        let offset = ((args.tile.batch * args.dims.m + row) * args.dims.n + col) as usize;
        unsafe {
            *out.get_unchecked_mut(offset) =
                args.base_scale * args.base[offset] + args.matmul_scale * acc;
        }
    }
}
