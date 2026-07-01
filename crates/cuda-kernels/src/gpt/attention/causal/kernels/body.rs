use cuda_device::{DisjointSlice, SharedArray, thread};

use crate::attention::CausalAttentionParams;
use crate::attention::layout::qkv_value;
use crate::block_reduce::block_sum_shared_f32_for_warps;
use crate::float_ptx::{exp_f32, fma_f32, ln_f32, safe_positive_denom};
use crate::warp_reduce::thread_lane_warp;

use super::score::{MAX_CAUSAL_TOKENS, ReduceCtx, score_denom, score_max};

pub(super) fn causal_attention_body<const WARPS: usize>(
    qkv: &[f32],
    out: &mut DisjointSlice<'_, f32>,
    log_sum_exp: &mut DisjointSlice<'_, f32>,
    params: CausalAttentionParams,
    scores: &mut SharedArray<f32, MAX_CAUSAL_TOKENS>,
    reduce: &mut SharedArray<f32, WARPS>,
) {
    let query = thread::blockIdx_x();
    let head = thread::blockIdx_y();
    let batch = thread::blockIdx_z();
    let (thread_index, lane, warp_in_block) = thread_lane_warp();
    let reduce_ctx = ReduceCtx::new(thread_index, lane, warp_in_block);

    let row = batch * params.seq_len + query;
    if query >= params.seq_len
        || head >= params.head_count
        || batch >= params.batch_size
        || row >= params.row_count
    {
        return;
    }

    let query_value = if thread_index < params.head_dim {
        qkv_value(qkv, batch, query, head, thread_index, 0, &params)
    } else {
        0.0
    };
    let mut key = 0;
    while key <= query {
        let local_dot = if thread_index < params.head_dim {
            query_value
                * qkv_value(
                    qkv,
                    batch,
                    key,
                    head,
                    thread_index,
                    params.embedding_dim,
                    &params,
                )
        } else {
            0.0
        };
        let dot = block_sum_shared_f32_for_warps(
            reduce,
            reduce_ctx.active_warps,
            local_dot,
            reduce_ctx.lane,
            reduce_ctx.warp_in_block,
        );
        if thread_index == 0 {
            scores[key as usize] = dot * params.scale;
        }

        thread::sync_threads();
        key += 1;
    }

    let score_max = score_max(query, reduce_ctx, scores, reduce);
    let denom = safe_positive_denom(score_denom(query, reduce_ctx, score_max, scores, reduce));
    if thread_index == 0 {
        let log_sum_exp_index = (batch as usize * params.head_count as usize + head as usize)
            * params.seq_len as usize
            + query as usize;
        unsafe {
            *log_sum_exp.get_unchecked_mut(log_sum_exp_index) = score_max + ln_f32(denom);
        }
    }

    if thread_index < params.head_dim {
        let mut value = 0.0;
        key = 0;
        while key <= query {
            let score = scores[key as usize];
            let weight = exp_f32(score - score_max) / denom;
            value = fma_f32(
                weight,
                qkv_value(
                    qkv,
                    batch,
                    key,
                    head,
                    thread_index,
                    params.embedding_dim * 2,
                    &params,
                ),
                value,
            );
            key += 1;
        }

        let out_index = row as usize * params.embedding_dim as usize
            + head as usize * params.head_dim as usize
            + thread_index as usize;
        unsafe {
            *out.get_unchecked_mut(out_index) = value;
        }
    }
}
