use cuda_device::{DisjointSlice, thread};

use super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use crate::attention::CausalAttentionParams;
use crate::attention::layout::{batched_qkv_index, compact_linear_parts, row_index};
use crate::float_ptx::{exp_f32, fma_f32, sincos_f32};

pub(super) fn scatter_body(
    d_q: &[f32],
    d_k: &[f32],
    d_v: &[f32],
    mut d_qkv: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let index = thread::blockIdx_x() * TC_BACKWARD_THREADS_PER_BLOCK + thread::threadIdx_x();
    let total = params.batch_size * params.head_count * params.seq_len * params.head_dim;
    if index >= total {
        return;
    }

    let (dim, token, _bh, batch, head) = compact_linear_parts(index, &params);
    let row = row_index(batch, token, &params);
    let q = batched_qkv_index(batch, token, head, dim, 0, &params);
    let k = batched_qkv_index(batch, token, head, dim, params.embedding_dim, &params);
    let v = batched_qkv_index(batch, token, head, dim, params.embedding_dim * 2, &params);
    if row >= params.row_count {
        unsafe {
            *d_qkv.get_unchecked_mut(q) = 0.0;
            *d_qkv.get_unchecked_mut(k) = 0.0;
            *d_qkv.get_unchecked_mut(v) = 0.0;
        }
        return;
    }

    let pair_index = index ^ 1;
    let dq = rope_raw_grad(
        token,
        dim,
        d_q[index as usize] * params.scale,
        d_q[pair_index as usize] * params.scale,
        params.head_dim,
    );
    let dk = rope_raw_grad(
        token,
        dim,
        d_k[index as usize] * params.scale,
        d_k[pair_index as usize] * params.scale,
        params.head_dim,
    );

    unsafe {
        *d_qkv.get_unchecked_mut(q) = dq;
        *d_qkv.get_unchecked_mut(k) = dk;
        *d_qkv.get_unchecked_mut(v) = d_v[index as usize];
    }
}

#[inline(always)]
fn rope_raw_grad(token: u32, dim: u32, grad_dim: f32, grad_pair: f32, head_dim: u32) -> f32 {
    let (sin, cos) = sincos_f32(token as f32 * rope_inv_freq(dim, head_dim));
    if dim & 1 == 0 {
        fma_f32(grad_pair, sin, grad_dim * cos)
    } else {
        fma_f32(-grad_pair, sin, grad_dim * cos)
    }
}

#[inline(always)]
fn rope_inv_freq(dim: u32, head_dim: u32) -> f32 {
    exp_f32(-9.210_340_5 * (dim & !1) as f32 / head_dim as f32)
}
