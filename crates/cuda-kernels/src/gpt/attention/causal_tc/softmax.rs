use cuda_device::{DisjointSlice, SharedArray, thread};

use super::gather::TC_FORWARD_THREADS_PER_BLOCK;
use crate::attention::CausalAttentionParams;
use crate::float_ptx::{exp_f32, ln_f32, max_f32};

mod index;
mod reduce;

use index::{log_sum_exp_index, score, score_index};
use reduce::{block_reduce_max, block_reduce_sum};

pub(super) const WARPS_PER_BLOCK: usize = (TC_FORWARD_THREADS_PER_BLOCK / 32) as usize;
pub(super) const NEG_INFINITY: f32 = -3.4028235e38_f32;

pub(super) fn softmax_body(
    scores: &[f32],
    mut probs: DisjointSlice<f32>,
    mut log_sum_exp: DisjointSlice<f32>,
    params: CausalAttentionParams,
    reduce: &mut SharedArray<f32, WARPS_PER_BLOCK>,
) {
    let query = thread::blockIdx_x();
    let head = thread::blockIdx_y();
    let batch = thread::blockIdx_z();
    let tid = thread::threadIdx_x();
    let row = batch * params.seq_len + query;
    if row >= params.row_count {
        return;
    }

    let max_score = query_max(scores, query, head, batch, tid, params, reduce);
    let denom = query_denom(scores, query, head, batch, tid, params, max_score, reduce);
    if tid == 0 {
        unsafe {
            *log_sum_exp.get_unchecked_mut(log_sum_exp_index(batch, query, head, &params)) =
                max_score + ln_f32(denom);
        }
    }

    let mut key = tid;
    while key < params.seq_len {
        let prob = if key <= query {
            exp_f32(score(scores, batch, head, query, key, &params) - max_score) / denom
        } else {
            0.0
        };
        unsafe {
            *probs.get_unchecked_mut(score_index(batch, head, query, key, &params)) = prob;
        }
        key += TC_FORWARD_THREADS_PER_BLOCK;
    }
}

fn query_max(
    scores: &[f32],
    query: u32,
    head: u32,
    batch: u32,
    tid: u32,
    params: CausalAttentionParams,
    reduce: &mut SharedArray<f32, WARPS_PER_BLOCK>,
) -> f32 {
    let mut local = NEG_INFINITY;
    let mut key = tid;
    while key <= query {
        local = max_f32(local, score(scores, batch, head, query, key, &params));
        key += TC_FORWARD_THREADS_PER_BLOCK;
    }
    block_reduce_max(local, tid, reduce)
}

fn query_denom(
    scores: &[f32],
    query: u32,
    head: u32,
    batch: u32,
    tid: u32,
    params: CausalAttentionParams,
    max_score: f32,
    reduce: &mut SharedArray<f32, WARPS_PER_BLOCK>,
) -> f32 {
    let mut local = 0.0;
    let mut key = tid;
    while key <= query {
        local += exp_f32(score(scores, batch, head, query, key, &params) - max_score);
        key += TC_FORWARD_THREADS_PER_BLOCK;
    }
    block_reduce_sum(local, tid, reduce)
}
