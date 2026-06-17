use cuda_device::SharedArray;

use super::dkv_thread::KeyThread;
use super::layout::{softmax_d_value, softmax_prob};
use super::types::{CAUSAL_BACKWARD_KEY_BLOCK, CausalAttentionBackwardParams};

#[inline(always)]
pub(super) fn score_local(query_value: f32, active: bool, key_value: f32) -> f32 {
    if active { query_value * key_value } else { 0.0 }
}

#[inline(always)]
pub(super) fn dp_local(d_out_query: f32, active: bool, value_value: f32) -> f32 {
    if active {
        d_out_query * value_value
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
    active: bool,
    score_dp: (f32, f32),
    prob: &mut SharedArray<f32, { CAUSAL_BACKWARD_KEY_BLOCK as usize }>,
    ds: &mut SharedArray<f32, { CAUSAL_BACKWARD_KEY_BLOCK as usize }>,
) {
    let p = if active {
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
