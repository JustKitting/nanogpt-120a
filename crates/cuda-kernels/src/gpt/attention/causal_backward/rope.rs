use crate::float_ptx::{exp_f32, fma_f32, sincos_f32};

use super::layout::qkv_index;
use super::types::CausalAttentionBackwardParams;

#[inline(always)]
pub(super) fn q_value(
    qkv: &[f32],
    token: u32,
    head: u32,
    dim: u32,
    params: &CausalAttentionBackwardParams,
) -> f32 {
    rope_value(qkv, token, head, dim, 0, params)
}

#[inline(always)]
pub(super) fn k_value(
    qkv: &[f32],
    token: u32,
    head: u32,
    dim: u32,
    params: &CausalAttentionBackwardParams,
) -> f32 {
    rope_value(qkv, token, head, dim, params.embedding_dim, params)
}

#[inline(always)]
pub(super) fn rope_raw_grad(
    token: u32,
    dim: u32,
    grad_dim: f32,
    grad_pair: f32,
    head_dim: u32,
) -> f32 {
    let (sin, cos) = sincos_f32(token as f32 * rope_inv_freq(dim, head_dim));
    if dim & 1 == 0 {
        fma_f32(grad_pair, sin, grad_dim * cos)
    } else {
        fma_f32(-grad_pair, sin, grad_dim * cos)
    }
}

#[inline(always)]
fn rope_value(
    qkv: &[f32],
    token: u32,
    head: u32,
    dim: u32,
    section_offset: u32,
    params: &CausalAttentionBackwardParams,
) -> f32 {
    let value = qkv[qkv_index(token, head, dim, section_offset, params)];
    let paired = qkv[qkv_index(token, head, dim ^ 1, section_offset, params)];
    let (sin, cos) = sincos_f32(token as f32 * rope_inv_freq(dim, params.head_dim));

    if dim & 1 == 0 {
        fma_f32(-paired, sin, value * cos)
    } else {
        fma_f32(paired, sin, value * cos)
    }
}

#[inline(always)]
fn rope_inv_freq(dim: u32, head_dim: u32) -> f32 {
    exp_f32(-9.210_340_5 * (dim & !1) as f32 / head_dim as f32)
}
