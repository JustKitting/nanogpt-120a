use cuda_device::thread;

use crate::attention::CausalAttentionParams;
use crate::float_ptx::{exp_f32, ln_f32};

pub(crate) const KDA_MAX_HEAD_DIM: usize = 64;
pub(crate) const KDA_STATE_ELEMS: usize = KDA_MAX_HEAD_DIM * KDA_MAX_HEAD_DIM;
pub(crate) const KDA_MATRIX_ELEMS: usize = KDA_MAX_HEAD_DIM * KDA_MAX_HEAD_DIM;
pub(crate) const KDA_DENOM_EPS: f32 = 1.0e-6;
const KDA_DECAY_EXP_LIMIT: f32 = 3.0;

#[inline(always)]
pub(crate) fn kda_tc_shape(params: &CausalAttentionParams) -> bool {
    params.head_dim == KDA_MAX_HEAD_DIM as u32 && params.chunk_size == KDA_MAX_HEAD_DIM as u32
}

#[inline(always)]
pub(crate) fn compact_elems(params: &CausalAttentionParams) -> u32 {
    params.batch_size * params.head_count * params.seq_len * params.head_dim
}

#[inline(always)]
pub(crate) fn batch_head(params: &CausalAttentionParams) -> u32 {
    params.batch_size * params.head_count
}

#[inline(always)]
pub(crate) fn chunk_count(params: &CausalAttentionParams) -> u32 {
    params.seq_len.div_ceil(params.chunk_size)
}

#[inline(always)]
pub(crate) fn chunk_g_last_elems(params: &CausalAttentionParams) -> u32 {
    batch_head(params) * chunk_count(params) * params.head_dim
}

#[inline(always)]
pub(crate) fn chunk_matrix_elems(params: &CausalAttentionParams) -> u32 {
    params.chunk_size * params.chunk_size
}

#[inline(always)]
pub(crate) fn state_elems(params: &CausalAttentionParams) -> u32 {
    params.head_dim * params.head_dim
}

#[inline(always)]
pub(crate) fn linear_thread_index(threads_per_block: u32, total: u32) -> Option<u32> {
    let index = thread::blockIdx_x() * threads_per_block + thread::threadIdx_x();
    if index < total { Some(index) } else { None }
}

#[inline(always)]
pub(crate) fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline(always)]
pub(crate) fn silu_grad(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

#[inline(always)]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp_f32(-x))
}

#[inline(always)]
pub(crate) fn safe_denom(x: f32) -> f32 {
    if x >= 0.0 {
        x + KDA_DENOM_EPS
    } else {
        x - KDA_DENOM_EPS
    }
}

#[inline(always)]
pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        ln_f32(1.0 + exp_f32(x))
    }
}

#[inline(always)]
pub(crate) fn kda_decay_exp(x: f32) -> f32 {
    exp_f32(if x > KDA_DECAY_EXP_LIMIT {
        KDA_DECAY_EXP_LIMIT
    } else if x < -KDA_DECAY_EXP_LIMIT {
        -KDA_DECAY_EXP_LIMIT
    } else {
        x
    })
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

#[inline(always)]
pub(crate) fn q_offset(_params: &CausalAttentionParams) -> u32 {
    0
}

#[inline(always)]
pub(crate) fn k_offset(params: &CausalAttentionParams) -> u32 {
    params.embedding_dim
}

#[inline(always)]
pub(crate) fn v_offset(params: &CausalAttentionParams) -> u32 {
    params.embedding_dim * 2
}

#[inline(always)]
pub(crate) fn g_offset(params: &CausalAttentionParams) -> u32 {
    params.embedding_dim * 3
}

#[inline(always)]
pub(crate) fn beta_offset(params: &CausalAttentionParams) -> u32 {
    params.embedding_dim * 4
}
