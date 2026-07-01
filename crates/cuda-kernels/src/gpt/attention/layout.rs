use super::causal::CausalAttentionParams;
use super::rope::ApplyRopeParams;

#[inline(always)]
pub(crate) fn row_index(batch: u32, token: u32, params: &CausalAttentionParams) -> u32 {
    batch * params.seq_len + token
}

#[inline(always)]
pub(crate) fn compact_index(
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    params: &CausalAttentionParams,
) -> usize {
    (((batch * params.head_count + head) * params.seq_len + token) * params.head_dim + dim) as usize
}

#[inline(always)]
pub(crate) fn compact_linear_parts(
    index: u32,
    params: &CausalAttentionParams,
) -> (u32, u32, u32, u32, u32) {
    let dim = index % params.head_dim;
    let token = (index / params.head_dim) % params.seq_len;
    let bh = index / (params.seq_len * params.head_dim);
    let batch = bh / params.head_count;
    let head = bh - batch * params.head_count;
    (dim, token, bh, batch, head)
}

#[inline(always)]
pub(crate) fn hidden_index(
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    params: &CausalAttentionParams,
) -> usize {
    row_index(batch, token, params) as usize * params.embedding_dim as usize
        + head as usize * params.head_dim as usize
        + dim as usize
}

#[inline(always)]
pub(crate) fn qkv_index(
    row: u32,
    head: u32,
    dim: u32,
    section_offset: u32,
    params: &CausalAttentionParams,
) -> usize {
    qkv_index_from_shape(
        row,
        head,
        dim,
        section_offset,
        params.qkv_dim,
        params.head_dim,
    )
}

#[inline(always)]
pub(crate) fn batched_qkv_index(
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    section_offset: u32,
    params: &CausalAttentionParams,
) -> usize {
    qkv_index(
        row_index(batch, token, params),
        head,
        dim,
        section_offset,
        params,
    )
}

#[inline(always)]
pub(crate) fn qkv_value<T: Copy>(
    qkv: &[T],
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    section_offset: u32,
    params: &CausalAttentionParams,
) -> T {
    qkv[batched_qkv_index(batch, token, head, dim, section_offset, params)]
}

#[inline(always)]
pub(crate) fn rope_qkv_index(
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    section_offset: u32,
    params: &ApplyRopeParams,
) -> usize {
    let row = batch * params.seq_len + token;
    qkv_index_from_shape(
        row,
        head,
        dim,
        section_offset,
        params.qkv_dim,
        params.head_dim,
    )
}

#[inline(always)]
fn qkv_index_from_shape(
    row: u32,
    head: u32,
    dim: u32,
    section_offset: u32,
    qkv_dim: u32,
    head_dim: u32,
) -> usize {
    row as usize * qkv_dim as usize
        + section_offset as usize
        + head as usize * head_dim as usize
        + dim as usize
}
