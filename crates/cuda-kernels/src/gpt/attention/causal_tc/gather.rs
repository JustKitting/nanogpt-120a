use cuda_device::{DisjointSlice, thread};

use crate::attention::CausalAttentionParams;
use crate::attention::layout::{batched_qkv_index, compact_linear_parts, row_index};

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

    let (dim, token, _bh, batch, head) = compact_linear_parts(index, &params);
    let row = row_index(batch, token, &params);
    if row >= params.row_count {
        unsafe {
            *q.get_unchecked_mut(index as usize) = 0.0;
            *k.get_unchecked_mut(index as usize) = 0.0;
            *v.get_unchecked_mut(index as usize) = 0.0;
        }
        return;
    }

    unsafe {
        *q.get_unchecked_mut(index as usize) =
            qkv[batched_qkv_index(batch, token, head, dim, 0, &params)];
        *k.get_unchecked_mut(index as usize) =
            qkv[batched_qkv_index(batch, token, head, dim, params.embedding_dim, &params)];
        *v.get_unchecked_mut(index as usize) =
            qkv[batched_qkv_index(batch, token, head, dim, params.embedding_dim * 2, &params)];
    }
}
