use cuda_device::SharedArray;

use super::dkv_thread::KeyThread;
use super::layout::{d_out_value, softmax_d_value, softmax_prob, v_value};
use super::rope::{k_value, q_value};
use super::types::{CAUSAL_BACKWARD_KEY_BLOCK, CausalAttentionBackwardParams};

#[inline(always)]
pub(super) fn score_local(
    qkv: &[f32],
    params: &CausalAttentionBackwardParams,
    thread_state: KeyThread,
    query: u32,
    active: bool,
) -> f32 {
    if active {
        q_value(
            qkv,
            thread_state.batch,
            query,
            thread_state.head,
            thread_state.dim,
            params,
        ) * k_value(
            qkv,
            thread_state.batch,
            thread_state.key(),
            thread_state.head,
            thread_state.dim,
            params,
        )
    } else {
        0.0
    }
}

#[inline(always)]
pub(super) fn dp_local(
    qkv: &[f32],
    d_out: &[f32],
    params: &CausalAttentionBackwardParams,
    thread_state: KeyThread,
    query: u32,
    active: bool,
) -> f32 {
    if active {
        d_out_value(
            d_out,
            thread_state.batch,
            query,
            thread_state.head,
            thread_state.dim,
            params,
        ) * v_value(
            qkv,
            thread_state.batch,
            thread_state.key(),
            thread_state.head,
            thread_state.dim,
            params,
        )
    } else {
        0.0
    }
}

#[inline(always)]
pub(super) fn write_query_scalars(
    log_sum_exp: &[f32],
    softmax_d: &[f32],
    params: &CausalAttentionBackwardParams,
    thread_state: KeyThread,
    query: u32,
    score_dp: (f32, f32),
    prob: &mut SharedArray<f32, { CAUSAL_BACKWARD_KEY_BLOCK as usize }>,
    ds: &mut SharedArray<f32, { CAUSAL_BACKWARD_KEY_BLOCK as usize }>,
) {
    let p = if thread_state.active(query, params) {
        softmax_prob(
            score_dp.0,
            thread_state.batch,
            query,
            thread_state.head,
            log_sum_exp,
            params,
        )
    } else {
        0.0
    };
    prob[thread_state.key_offset as usize] = p;
    ds[thread_state.key_offset as usize] = p
        * (score_dp.1
            - softmax_d_value(
                softmax_d,
                thread_state.batch,
                query,
                thread_state.head,
                params,
            ));
}
