use crate::attention::CausalAttentionParams;
pub(crate) use crate::attention::layout::{
    compact_index, compact_linear_parts, hidden_index, qkv_index,
};

use super::activation::beta_offset;
use super::shape::{chunk_count, state_elems};

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
pub(crate) fn beta_index(row: u32, head: u32, params: &CausalAttentionParams) -> usize {
    row as usize * params.qkv_dim as usize + beta_offset(params) as usize + head as usize
}
