use cuda_device::{SharedArray, thread};

use super::dkv_scalars::{dp_local, score_local, write_query_scalars};
use super::dkv_thread::KeyThread;
use super::layout::d_out_value;
use super::reductions::reduce_key;
use super::rope::q_value;
use super::types::{CAUSAL_BACKWARD_KEY_BLOCK, CausalAttentionBackwardParams};

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn accumulate_key(
    qkv: &[f32],
    d_out: &[f32],
    log_sum_exp: &[f32],
    softmax_d: &[f32],
    params: &CausalAttentionBackwardParams,
    thread_state: KeyThread,
    reduce: &mut SharedArray<f32, { (CAUSAL_BACKWARD_KEY_BLOCK * 2) as usize }>,
    prob: &mut SharedArray<f32, { CAUSAL_BACKWARD_KEY_BLOCK as usize }>,
    ds: &mut SharedArray<f32, { CAUSAL_BACKWARD_KEY_BLOCK as usize }>,
) -> (f32, f32) {
    let mut dk_rot = 0.0;
    let mut dv = 0.0;
    let mut query = thread_state.key();
    while query < params.seq_len {
        let active = thread_state.active(query, params);
        let score = reduce_key(
            score_local(qkv, params, thread_state, query, active),
            thread_state.key_offset,
            thread_state.lane,
            thread_state.warp_in_key,
            reduce,
        );
        let dp = reduce_key(
            dp_local(qkv, d_out, params, thread_state, query, active),
            thread_state.key_offset,
            thread_state.lane,
            thread_state.warp_in_key,
            reduce,
        );

        if thread_state.dim == 0 {
            write_query_scalars(
                log_sum_exp,
                softmax_d,
                params,
                thread_state,
                query,
                (score, dp),
                prob,
                ds,
            );
        }
        thread::sync_threads();

        if active {
            let p = prob[thread_state.key_offset as usize];
            let d_score = ds[thread_state.key_offset as usize];
            dv += p * d_out_value(
                d_out,
                thread_state.batch,
                query,
                thread_state.head,
                thread_state.dim,
                params,
            );
            dk_rot += d_score
                * q_value(
                    qkv,
                    thread_state.batch,
                    query,
                    thread_state.head,
                    thread_state.dim,
                    params,
                )
                * params.scale;
        }
        query += 1;
    }
    (dk_rot, dv)
}
