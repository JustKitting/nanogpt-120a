use cuda_device::{DisjointSlice, SharedArray, thread, warp};

use super::layout::{d_out_value, hidden_value};
use super::reductions::reduce_head;
use super::types::CausalAttentionBackwardParams;

pub(super) fn softmax_d_body(
    out: &[f32],
    d_out: &[f32],
    mut softmax_d: DisjointSlice<f32>,
    params: CausalAttentionBackwardParams,
    reduce: &mut SharedArray<f32, 2>,
) {
    let token = thread::blockIdx_x();
    let head = thread::blockIdx_y();
    let dim = thread::threadIdx_x();
    let lane = warp::lane_id();
    let local = if dim < params.head_dim {
        hidden_value(out, token, head, dim, &params) * d_out_value(d_out, token, head, dim, &params)
    } else {
        0.0
    };
    let value = reduce_head(local, lane, dim / 32, reduce);

    if dim == 0 {
        let index = head as usize * params.token_count as usize + token as usize;
        unsafe {
            *softmax_d.get_unchecked_mut(index) = value;
        }
    }
}
