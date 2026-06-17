use cuda_device::{SharedArray, thread};

use crate::warp_reduce::warp_sum_f32;

use super::types::{CAUSAL_BACKWARD_HEAD_DIM_THREADS, CAUSAL_BACKWARD_KEY_BLOCK};

const WARPS_PER_HEAD: u32 = CAUSAL_BACKWARD_HEAD_DIM_THREADS / 32;
const KEY_REDUCE_LEN: usize = (CAUSAL_BACKWARD_KEY_BLOCK * WARPS_PER_HEAD) as usize;
pub(super) const KEY_REDUCE_PAIR_LEN: usize = KEY_REDUCE_LEN * 2;

#[inline(always)]
pub(super) fn reduce_head(
    local: f32,
    lane: u32,
    warp_in_head: u32,
    shared: &mut SharedArray<f32, { WARPS_PER_HEAD as usize }>,
) -> f32 {
    let warp_total = warp_sum_f32(local);
    if lane == 0 {
        shared[warp_in_head as usize] = warp_total;
    }
    thread::sync_threads();

    if warp_in_head == 0 && lane == 0 {
        let mut total = 0.0;
        let mut warp = 0;
        while warp < WARPS_PER_HEAD {
            total += shared[warp as usize];
            warp += 1;
        }
        shared[0] = total;
    }
    thread::sync_threads();
    shared[0]
}

#[inline(always)]
pub(super) fn reduce_key_pair(
    local_a: f32,
    local_b: f32,
    key_offset: u32,
    lane: u32,
    warp_in_key: u32,
    shared: &mut SharedArray<f32, KEY_REDUCE_PAIR_LEN>,
) -> (f32, f32) {
    let base_a = key_offset * WARPS_PER_HEAD;
    let base_b = base_a + KEY_REDUCE_LEN as u32;
    let warp_a = warp_sum_f32(local_a);
    let warp_b = warp_sum_f32(local_b);
    if lane == 0 {
        shared[(base_a + warp_in_key) as usize] = warp_a;
        shared[(base_b + warp_in_key) as usize] = warp_b;
    }
    thread::sync_threads();

    if warp_in_key == 0 && lane == 0 {
        let mut total_a = 0.0;
        let mut total_b = 0.0;
        let mut warp = 0;
        while warp < WARPS_PER_HEAD {
            total_a += shared[(base_a + warp) as usize];
            total_b += shared[(base_b + warp) as usize];
            warp += 1;
        }
        shared[base_a as usize] = total_a;
        shared[base_b as usize] = total_b;
    }
    thread::sync_threads();
    (shared[base_a as usize], shared[base_b as usize])
}
