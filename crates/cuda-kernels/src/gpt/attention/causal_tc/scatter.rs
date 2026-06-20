use cuda_device::{DisjointSlice, thread};

use super::gather::TC_FORWARD_THREADS_PER_BLOCK;
use crate::attention::CausalAttentionParams;

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

    let dim = index % params.head_dim;
    let token = (index / params.head_dim) % params.seq_len;
    let batch_head = index / (params.seq_len * params.head_dim);
    let batch = batch_head / params.head_count;
    let head = batch_head - batch * params.head_count;
    let row = batch * params.seq_len + token;
    if row >= params.row_count {
        return;
    }

    let out_index = (row as usize * params.embedding_dim as usize)
        + head as usize * params.head_dim as usize
        + dim as usize;
    unsafe {
        *out.get_unchecked_mut(out_index) = compact[index as usize];
    }
}
