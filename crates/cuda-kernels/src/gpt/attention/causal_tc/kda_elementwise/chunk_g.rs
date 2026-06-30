use cuda_device::DisjointSlice;

use super::thread_index;
use crate::attention::CausalAttentionParams;
use crate::kda_common::{chunk_count, chunk_end_token, chunk_g_last_elems, compact_index};
use crate::kda_elementwise::chunk_cumsum_g_body as shared_chunk_cumsum_g_body;

pub(in super::super) fn chunk_cumsum_g_body(g: DisjointSlice<f32>, params: CausalAttentionParams) {
    shared_chunk_cumsum_g_body(g, params);
}

pub(in super::super) fn store_chunk_g_last_body(
    g: &[f32],
    mut chunk_g_last: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let Some(index) = thread_index(chunk_g_last_elems(&params)) else {
        return;
    };
    let chunks = chunk_count(&params);
    let dim = index % params.head_dim;
    let chunk = (index / params.head_dim) % chunks;
    let bh = index / (chunks * params.head_dim);
    let batch = bh / params.head_count;
    let head = bh - batch * params.head_count;
    let chunk_end = chunk_end_token(chunk, &params);
    unsafe {
        *chunk_g_last.get_unchecked_mut(index as usize) =
            g[compact_index(batch, chunk_end, head, dim, &params)];
    }
}
