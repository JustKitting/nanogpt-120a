use cuda_device::{DisjointSlice, SharedArray, thread, warp};

use crate::attention::CausalAttentionParams;
use crate::attention::layout::{hidden_index, row_index};
use crate::f16_tc_matmul::convert::cvt_f32_f16;
use crate::warp_reduce::warp_sum_f32;

#[inline(always)]
fn value(
    values: &[f32],
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    params: &CausalAttentionParams,
) -> f32 {
    values[hidden_index(batch, token, head, dim, params)]
}

#[inline(always)]
fn reduce_head(local: f32, lane: u32, warp_in_head: u32, shared: &mut SharedArray<f32, 2>) -> f32 {
    let warp_total = warp_sum_f32(local);
    if lane == 0 {
        shared[warp_in_head as usize] = warp_total;
    }
    thread::sync_threads();

    if warp_in_head == 0 && lane == 0 {
        shared[0] += shared[1];
    }
    thread::sync_threads();
    shared[0]
}

pub(super) fn softmax_d_f16_body(
    out: &[u16],
    d_out: &[f32],
    mut softmax_d: DisjointSlice<f32>,
    params: CausalAttentionParams,
    reduce: &mut SharedArray<f32, 2>,
) {
    let token = thread::blockIdx_x();
    let head = thread::blockIdx_y();
    let batch = thread::blockIdx_z();
    let dim = thread::threadIdx_x();
    let lane = warp::lane_id();
    let row = row_index(batch, token, &params);
    if token >= params.seq_len || head >= params.head_count || row >= params.row_count {
        return;
    }

    let local = if dim < params.head_dim {
        value_f16(out, batch, token, head, dim, &params)
            * value(d_out, batch, token, head, dim, &params)
    } else {
        0.0
    };
    let sum = reduce_head(local, lane, dim / 32, reduce);

    if dim == 0 {
        let index = (batch as usize * params.head_count as usize + head as usize)
            * params.seq_len as usize
            + token as usize;
        unsafe {
            *softmax_d.get_unchecked_mut(index) = sum;
        }
    }
}

#[inline(always)]
fn value_f16(
    values: &[u16],
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    params: &CausalAttentionParams,
) -> f32 {
    cvt_f32_f16(values[hidden_index(batch, token, head, dim, params)])
}
