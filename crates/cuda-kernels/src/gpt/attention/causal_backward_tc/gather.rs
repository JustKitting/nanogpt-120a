use cuda_device::{DisjointSlice, thread};

use super::types::CausalAttentionBackwardTcParams;
use crate::f16_tc_matmul::convert::cvt_f32_f16;

pub(super) const TC_BACKWARD_THREADS_PER_BLOCK: u32 = 256;

#[allow(clippy::too_many_arguments)]
pub(super) fn gather_body(
    qkv: &[u16],
    d_out_src: &[f32],
    mut q: DisjointSlice<f32>,
    mut k: DisjointSlice<f32>,
    mut v: DisjointSlice<f32>,
    mut d_out: DisjointSlice<f32>,
    params: CausalAttentionBackwardTcParams,
) {
    let index = thread::blockIdx_x() * TC_BACKWARD_THREADS_PER_BLOCK + thread::threadIdx_x();
    let total = params.batch_size * params.head_count * params.seq_len * params.head_dim;
    if index >= total {
        return;
    }

    let dim = index % params.head_dim;
    let token = (index / params.head_dim) % params.seq_len;
    let batch_head = index / (params.seq_len * params.head_dim);
    let batch = batch_head / params.head_count;
    let head = batch_head - batch * params.head_count;
    let row = batch * params.seq_len + token;
    if row >= params.row_count {
        return;
    }

    unsafe {
        *q.get_unchecked_mut(index as usize) = qkv_value(qkv, batch, token, head, dim, 0, &params);
        *k.get_unchecked_mut(index as usize) =
            qkv_value(qkv, batch, token, head, dim, params.embedding_dim, &params);
        *v.get_unchecked_mut(index as usize) = qkv_value(
            qkv,
            batch,
            token,
            head,
            dim,
            params.embedding_dim * 2,
            &params,
        );
        *d_out.get_unchecked_mut(index as usize) =
            d_out_src[hidden_index(batch, token, head, dim, &params)];
    }
}

#[inline(always)]
fn qkv_value(
    qkv: &[u16],
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    section_offset: u32,
    params: &CausalAttentionBackwardTcParams,
) -> f32 {
    cvt_f32_f16(qkv[qkv_index(batch, token, head, dim, section_offset, params)])
}

#[inline(always)]
pub(super) fn qkv_index(
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    section_offset: u32,
    params: &CausalAttentionBackwardTcParams,
) -> usize {
    (batch as usize * params.seq_len as usize + token as usize) * params.qkv_dim as usize
        + section_offset as usize
        + head as usize * params.head_dim as usize
        + dim as usize
}

#[inline(always)]
fn hidden_index(
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    params: &CausalAttentionBackwardTcParams,
) -> usize {
    (batch as usize * params.seq_len as usize + token as usize) * params.embedding_dim as usize
        + head as usize * params.head_dim as usize
        + dim as usize
}
