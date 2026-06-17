use cuda_device::{SharedArray, thread};

use super::dkv_scalars::{dp_local, score_local, write_query_scalars};
use super::dkv_thread::KeyThread;
use super::layout::{d_out_value, v_value};
use super::reductions::reduce_key;
use super::rope::{k_value, q_value};
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
    let key = thread_state.key();
    let key_valid = thread_state.valid(params);
    let key_value = if key_valid {
        k_value(
            qkv,
            thread_state.batch,
            key,
            thread_state.head,
            thread_state.dim,
            params,
        )
    } else {
        0.0
    };
    let value_value = if key_valid {
        v_value(
            qkv,
            thread_state.batch,
            key,
            thread_state.head,
            thread_state.dim,
            params,
        )
    } else {
        0.0
    };
    let mut query = key;
    while query < params.seq_len {
        let active = thread_state.active(query, params);
        let query_value = if active {
            q_value(
                qkv,
                thread_state.batch,
                query,
                thread_state.head,
                thread_state.dim,
                params,
            )
        } else {
            0.0
        };
        let d_out_query = if active {
            d_out_value(
                d_out,
                thread_state.batch,
                query,
                thread_state.head,
                thread_state.dim,
                params,
            )
        } else {
            0.0
        };
        let score = reduce_key(
            score_local(query_value, active, key_value),
            thread_state.key_offset,
            thread_state.lane,
            thread_state.warp_in_key,
            reduce,
        );
        let dp = reduce_key(
            dp_local(d_out_query, active, value_value),
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
                active,
                (score, dp),
                prob,
                ds,
            );
        }
        thread::sync_threads();

        if active {
            let p = prob[thread_state.key_offset as usize];
            let d_score = ds[thread_state.key_offset as usize];
            dv += p * d_out_query;
            dk_rot += d_score * query_value * params.scale;
        }
        query += 1;
    }
    (dk_rot, dv)
}
