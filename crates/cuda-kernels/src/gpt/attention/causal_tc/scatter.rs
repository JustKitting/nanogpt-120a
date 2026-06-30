use cuda_device::{DisjointSlice, thread};

use super::gather::TC_FORWARD_THREADS_PER_BLOCK;
use crate::attention::CausalAttentionParams;
use crate::attention::layout::{compact_linear_parts, hidden_index, row_index};
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;

pub(super) fn scatter_output_body(
    compact: &[f32],
    mut out: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let index = thread::blockIdx_x() * TC_FORWARD_THREADS_PER_BLOCK + thread::threadIdx_x();
    let total = params.batch_size * params.head_count * params.seq_len * params.head_dim;
    if index >= total {
        return;
    }

    let (dim, token, _bh, batch, head) = compact_linear_parts(index, &params);
    let row = row_index(batch, token, &params);
    let out_index = hidden_index(batch, token, head, dim, &params);
    if row >= params.row_count {
        unsafe {
            *out.get_unchecked_mut(out_index) = 0.0;
        }
        return;
    }

    let value = compact[index as usize];
    unsafe {
        *out.get_unchecked_mut(out_index) = value;
    }
}

pub(super) fn scatter_output_save_f16_body(
    compact: &[f32],
    mut out: DisjointSlice<f32>,
    mut attention_out_f16: DisjointSlice<u16>,
    params: CausalAttentionParams,
) {
    let index = thread::blockIdx_x() * TC_FORWARD_THREADS_PER_BLOCK + thread::threadIdx_x();
    let total = params.batch_size * params.head_count * params.seq_len * params.head_dim;
    if index >= total {
        return;
    }

    let (dim, token, _bh, batch, head) = compact_linear_parts(index, &params);
    let row = row_index(batch, token, &params);
    let out_index = hidden_index(batch, token, head, dim, &params);
    if row >= params.row_count {
        unsafe {
            *out.get_unchecked_mut(out_index) = 0.0;
            *attention_out_f16.get_unchecked_mut(out_index) = 0;
        }
        return;
    }

    let value = compact[index as usize];
    unsafe {
        *out.get_unchecked_mut(out_index) = value;
        *attention_out_f16.get_unchecked_mut(out_index) = cvt_rn_f16_f32(value);
    }
}
