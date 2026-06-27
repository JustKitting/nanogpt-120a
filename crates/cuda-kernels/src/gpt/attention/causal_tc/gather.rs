use cuda_device::{DisjointSlice, thread};

use crate::attention::CausalAttentionParams;

pub(super) const TC_FORWARD_THREADS_PER_BLOCK: u32 = 256;

pub(super) fn gather_qkv_body(
    qkv: &[f32],
    mut q: DisjointSlice<f32>,
    mut k: DisjointSlice<f32>,
    mut v: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let index = thread::blockIdx_x() * TC_FORWARD_THREADS_PER_BLOCK + thread::threadIdx_x();
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
        unsafe {
            *q.get_unchecked_mut(index as usize) = 0.0;
            *k.get_unchecked_mut(index as usize) = 0.0;
            *v.get_unchecked_mut(index as usize) = 0.0;
        }
        return;
    }

    unsafe {
        *q.get_unchecked_mut(index as usize) = qkv[qkv_index(batch, token, head, dim, 0, &params)];
        *k.get_unchecked_mut(index as usize) =
            qkv[qkv_index(batch, token, head, dim, params.embedding_dim, &params)];
        *v.get_unchecked_mut(index as usize) =
            qkv[qkv_index(batch, token, head, dim, params.embedding_dim * 2, &params)];
    }
}

#[inline(always)]
fn qkv_index(
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    section_offset: u32,
    params: &CausalAttentionParams,
) -> usize {
    (batch as usize * params.seq_len as usize + token as usize) * params.qkv_dim as usize
        + section_offset as usize
        + head as usize * params.head_dim as usize
        + dim as usize
}
