use cuda_device::{SharedArray, grid, thread};

use crate::f16_tc_matmul::cta_tile::CTA_THREADS;
use crate::float_ptx::max_f32;

use super::super::super::super::threads::{WARP_SIZE, WARPS_PER_BLOCK};
use super::super::super::super::work_grid::WorkGrid;
use crate::device_ptr::read_f32;

const FP4_MAX: f32 = 6.0;
const FP8_MAX_FOUR_SIX: f32 = 256.0;
pub(super) fn reduce_global_scale(
    block_amax: *const f32,
    out_global_scale: *mut f32,
    warp_sums: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    work: WorkGrid,
) {
    if work.block() == 0 {
        let tid = thread::threadIdx_x();
        let lane = tid & (WARP_SIZE - 1);
        let warp_in_block = tid / WARP_SIZE;
        reduce_blocks_to_global_scale(
            tid,
            lane,
            warp_in_block,
            block_amax,
            out_global_scale,
            warp_sums,
            work,
        );
    }
    grid::sync();
}

fn reduce_blocks_to_global_scale(
    tid: u32,
    lane: u32,
    warp_in_block: u32,
    block_amax: *const f32,
    out_global_scale: *mut f32,
    warp_sums: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    work: WorkGrid,
) {
    let mut block = tid;
    let mut local_amax = 0.0;
    while block < work.blocks() {
        local_amax = max_f32(local_amax, read_f32(block_amax, block));
        block += CTA_THREADS;
    }
    let tensor_amax =
        crate::block_reduce::block_max_shared_f32(warp_sums, local_amax, lane, warp_in_block);
    if tid == 0 {
        let global_scale = if tensor_amax == 0.0 {
            1.0
        } else {
            tensor_amax / (FP8_MAX_FOUR_SIX * FP4_MAX)
        };
        unsafe {
            *out_global_scale = global_scale;
        }
    }
}
