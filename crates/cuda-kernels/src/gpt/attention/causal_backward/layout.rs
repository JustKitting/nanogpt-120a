use crate::float_ptx::exp_f32;

use super::types::CausalAttentionBackwardParams;

#[inline(always)]
pub(super) fn v_value(
    qkv: &[f32],
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    params: &CausalAttentionBackwardParams,
) -> f32 {
    qkv[qkv_index(batch, token, head, dim, params.embedding_dim * 2, params)]
}

#[inline(always)]
pub(super) fn d_out_value(
    d_out: &[f32],
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    params: &CausalAttentionBackwardParams,
) -> f32 {
    d_out[hidden_index(batch, token, head, dim, params)]
}

#[inline(always)]
pub(super) fn hidden_value(
    values: &[f32],
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    params: &CausalAttentionBackwardParams,
) -> f32 {
    values[hidden_index(batch, token, head, dim, params)]
}

#[inline(always)]
pub(super) fn log_sum_exp_value(
    log_sum_exp: &[f32],
    batch: u32,
    token: u32,
    head: u32,
    params: &CausalAttentionBackwardParams,
) -> f32 {
    log_sum_exp[(batch as usize * params.head_count as usize + head as usize)
        * params.seq_len as usize
        + token as usize]
}

#[inline(always)]
pub(super) fn softmax_d_value(
    softmax_d: &[f32],
    batch: u32,
    token: u32,
    head: u32,
    params: &CausalAttentionBackwardParams,
) -> f32 {
    softmax_d[(batch as usize * params.head_count as usize + head as usize)
        * params.seq_len as usize
        + token as usize]
}

#[inline(always)]
pub(super) fn qkv_index(
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    section_offset: u32,
    params: &CausalAttentionBackwardParams,
) -> usize {
    (batch as usize * params.seq_len as usize + token as usize) * params.qkv_dim as usize
        + section_offset as usize
        + head as usize * params.head_dim as usize
        + dim as usize
}

#[inline(always)]
pub(super) fn hidden_index(
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    params: &CausalAttentionBackwardParams,
) -> usize {
    (batch as usize * params.seq_len as usize + token as usize) * params.embedding_dim as usize
        + head as usize * params.head_dim as usize
        + dim as usize
}

#[inline(always)]
pub(super) fn softmax_prob(
    score: f32,
    batch: u32,
    token: u32,
    head: u32,
    log_sum_exp: &[f32],
    params: &CausalAttentionBackwardParams,
) -> f32 {
    exp_f32(score * params.scale - log_sum_exp_value(log_sum_exp, batch, token, head, params))
}
