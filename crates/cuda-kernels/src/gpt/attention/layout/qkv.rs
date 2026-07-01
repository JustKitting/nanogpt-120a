use super::super::causal::CausalAttentionParams;
use super::linear::row_index;

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
pub(super) fn qkv_index_from_shape(
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
