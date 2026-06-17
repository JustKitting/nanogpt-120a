use cuda_device::{DisjointSlice, SharedArray, thread, warp};

use super::layout::{d_out_value, qkv_index, softmax_d_value, softmax_prob, v_value};
use super::reductions::reduce_head;
use super::rope::{k_value, q_value, rope_raw_grad};
use super::types::{CAUSAL_BACKWARD_HEAD_DIM_THREADS, CausalAttentionBackwardParams};

pub(super) fn dq_body(
    qkv: &[f32],
    d_out: &[f32],
    lse: &[f32],
    softmax_d: &[f32],
    mut d_qkv: DisjointSlice<f32>,
    params: CausalAttentionBackwardParams,
    reduce: &mut SharedArray<f32, 2>,
    ds: &mut SharedArray<f32, 1>,
    dq_rot: &mut SharedArray<f32, { CAUSAL_BACKWARD_HEAD_DIM_THREADS as usize }>,
) {
    let query = thread::blockIdx_x();
    let head = thread::blockIdx_y();
    let dim = thread::threadIdx_x();
    let valid_dim = dim < params.head_dim;

    let lane = warp::lane_id();
    let warp_in_head = dim / 32;
    let mut grad = 0.0;
    let mut key = 0;
    while key <= query {
        let local_score = if valid_dim {
            q_value(qkv, query, head, dim, &params) * k_value(qkv, key, head, dim, &params)
        } else {
            0.0
        };
        let local_dp = if valid_dim {
            d_out_value(d_out, query, head, dim, &params) * v_value(qkv, key, head, dim, &params)
        } else {
            0.0
        };
        let score = reduce_head(local_score, lane, warp_in_head, reduce);
        let dp = reduce_head(local_dp, lane, warp_in_head, reduce);
        if dim == 0 {
            let p = softmax_prob(score, query, head, lse, &params);
            ds[0] = p * (dp - softmax_d_value(softmax_d, query, head, &params));
        }
        thread::sync_threads();
        if valid_dim {
            grad += ds[0] * k_value(qkv, key, head, dim, &params) * params.scale;
        }
        key += 1;
    }

    dq_rot[dim as usize] = grad;
    thread::sync_threads();

    if valid_dim {
        let value = rope_raw_grad(
            query,
            dim,
            grad,
            dq_rot[(dim ^ 1) as usize],
            params.head_dim,
        );
        unsafe {
            *d_qkv.get_unchecked_mut(qkv_index(query, head, dim, 0, &params)) = value;
        }
    }
}
