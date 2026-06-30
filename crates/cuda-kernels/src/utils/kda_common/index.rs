use crate::attention::CausalAttentionParams;

use super::activation::beta_offset;
use super::shape::{chunk_count, state_elems};

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
pub(crate) fn beta_compact_index(
    batch: u32,
    token: u32,
    head: u32,
    params: &CausalAttentionParams,
) -> usize {
    ((batch * params.head_count + head) * params.seq_len + token) as usize
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
    (batch as usize * params.seq_len as usize + token as usize) * params.embedding_dim as usize
        + head as usize * params.head_dim as usize
        + dim as usize
}

#[inline(always)]
pub(crate) fn chunk_matrix_index(
    bh: u32,
    chunk: u32,
    row: u32,
    col: u32,
    params: &CausalAttentionParams,
) -> usize {
    (((bh * chunk_count(params) + chunk) * params.chunk_size + row) * params.chunk_size + col)
        as usize
}

#[inline(always)]
pub(crate) fn chunk_state_index(
    bh: u32,
    chunk: u32,
    state_index: u32,
    params: &CausalAttentionParams,
) -> usize {
    ((bh * chunk_count(params) + chunk) * state_elems(params) + state_index) as usize
}

#[inline(always)]
pub(crate) fn chunk_g_last_index(
    bh: u32,
    chunk: u32,
    dim: u32,
    params: &CausalAttentionParams,
) -> usize {
    ((bh * chunk_count(params) + chunk) * params.head_dim + dim) as usize
}

#[inline(always)]
pub(crate) fn qkv_index(
    row: u32,
    head: u32,
    dim: u32,
    section_offset: u32,
    params: &CausalAttentionParams,
) -> usize {
    row as usize * params.qkv_dim as usize
        + section_offset as usize
        + head as usize * params.head_dim as usize
        + dim as usize
}

#[inline(always)]
pub(crate) fn beta_index(row: u32, head: u32, params: &CausalAttentionParams) -> usize {
    row as usize * params.qkv_dim as usize + beta_offset(params) as usize + head as usize
}
