use crate::f16_tc_matmul::convert::cvt_f32_f16;
use crate::float_ptx::{fma_f32, sqrt_f32};
use crate::kda_common::{KDA_DENOM_EPS, silu};

#[derive(Clone, Copy)]
pub(super) struct ClipParams {
    pub(super) qkv_dim: u32,
    pub(super) embedding_dim: u32,
    pub(super) head_dim: u32,
}

pub(super) fn qk_norms(
    qkv: &[u16],
    row: u32,
    head: u32,
    params: ClipParams,
    silu_qk: u32,
) -> (f32, f32) {
    let mut q_sum = 0.0;
    let mut k_sum = 0.0;
    let mut dim = 0;
    while dim < params.head_dim {
        let raw_q = cvt_f32_f16(qkv[qkv_index(row, head, dim, 0, params)]);
        let raw_k = cvt_f32_f16(qkv[qkv_index(row, head, dim, params.embedding_dim, params)]);
        let q = if silu_qk != 0 { silu(raw_q) } else { raw_q };
        let k = if silu_qk != 0 { silu(raw_k) } else { raw_k };
        q_sum = fma_f32(q, q, q_sum);
        k_sum = fma_f32(k, k, k_sum);
        dim += 1;
    }
    (
        sqrt_f32(q_sum + KDA_DENOM_EPS),
        sqrt_f32(k_sum + KDA_DENOM_EPS),
    )
}

fn qkv_index(row: u32, head: u32, dim: u32, section_offset: u32, params: ClipParams) -> usize {
    (row * params.qkv_dim + section_offset + head * params.head_dim + dim) as usize
}

#[allow(clippy::eq_op)]
pub(super) fn clip_factor(score: f32, tau: f32) -> f32 {
    if score == score && score > tau {
        sqrt_f32(tau / (score + KDA_DENOM_EPS))
    } else {
        1.0
    }
}
