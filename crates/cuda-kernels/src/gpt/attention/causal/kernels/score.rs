use cuda_device::{SharedArray, thread};

use crate::block_reduce::{block_max_shared_f32_for_warps, block_sum_shared_f32_for_warps};
use crate::float_ptx::{exp_f32, max_f32};

pub(super) const MAX_CAUSAL_TOKENS: usize = 1024;
const NEG_INFINITY: f32 = -3.4028235e38_f32;

#[derive(Clone, Copy)]
pub(super) struct ReduceCtx {
    pub(super) thread_index: u32,
    pub(super) lane: u32,
    pub(super) warp_in_block: u32,
    pub(super) active_warps: u32,
}

impl ReduceCtx {
    #[inline(always)]
    pub(super) fn new(thread_index: u32, lane: u32, warp_in_block: u32) -> Self {
        Self {
            thread_index,
            lane,
            warp_in_block,
            active_warps: thread::blockDim_x() / 32,
        }
    }
}

#[inline(always)]
pub(super) fn score_max<const WARPS: usize>(
    query: u32,
    reduce_ctx: ReduceCtx,
    scores: &SharedArray<f32, MAX_CAUSAL_TOKENS>,
    reduce: &mut SharedArray<f32, WARPS>,
) -> f32 {
    let mut local = NEG_INFINITY;
    let mut key = reduce_ctx.thread_index;
    while key <= query {
        local = max_f32(local, scores[key as usize]);
        key += thread::blockDim_x();
    }
    block_max_shared_f32_for_warps(
        reduce,
        reduce_ctx.active_warps,
        local,
        reduce_ctx.lane,
        reduce_ctx.warp_in_block,
        NEG_INFINITY,
    )
}

#[inline(always)]
pub(super) fn score_denom<const WARPS: usize>(
    query: u32,
    reduce_ctx: ReduceCtx,
    score_max: f32,
    scores: &SharedArray<f32, MAX_CAUSAL_TOKENS>,
    reduce: &mut SharedArray<f32, WARPS>,
) -> f32 {
    let mut local = 0.0;
    let mut key = reduce_ctx.thread_index;
    while key <= query {
        local += exp_f32(scores[key as usize] - score_max);
        key += thread::blockDim_x();
    }
    block_sum_shared_f32_for_warps(
        reduce,
        reduce_ctx.active_warps,
        local,
        reduce_ctx.lane,
        reduce_ctx.warp_in_block,
    )
}
