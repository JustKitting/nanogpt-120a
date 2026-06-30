use cuda_device::{DisjointSlice, SharedArray, thread};

use super::gather::TC_FORWARD_THREADS_PER_BLOCK;
use crate::attention::CausalAttentionParams;
use crate::float_ptx::fma_f32;
use crate::kda_common::{
    KDA_MATRIX_ELEMS, KDA_MAX_HEAD_DIM, batch_head, beta_compact_index, chunk_count,
    chunk_end_token, chunk_g_last_elems, chunk_g_last_index, chunk_matrix_elems, compact_elems,
    compact_index, compact_linear_parts, kda_decay_exp, linear_thread_index,
};
use crate::kda_elementwise::{
    chunk_cumsum_g_body as shared_chunk_cumsum_g_body, prepare_kda_inputs_body,
};

pub(super) fn prepare_kda_body(
    qkv: &[f32],
    q: DisjointSlice<f32>,
    k: DisjointSlice<f32>,
    v: DisjointSlice<f32>,
    g: DisjointSlice<f32>,
    beta: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    prepare_kda_inputs_body(qkv, q, k, v, g, beta, params, TC_FORWARD_THREADS_PER_BLOCK);
}

pub(super) fn zero_f32_body(mut values: DisjointSlice<f32>, element_count: u32) {
    let Some(index) = thread_index(element_count) else {
        return;
    };
    unsafe {
        *values.get_unchecked_mut(index as usize) = 0.0;
    }
}

pub(super) fn chunk_cumsum_g_body(g: DisjointSlice<f32>, params: CausalAttentionParams) {
    shared_chunk_cumsum_g_body(g, params);
}

pub(super) fn make_qg_kneg_body(
    mut q: DisjointSlice<f32>,
    k: &[f32],
    g: &[f32],
    mut kneg: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let Some(index) = thread_index(compact_elems(&params)) else {
        return;
    };
    let value_g = g[index as usize];
    unsafe {
        *q.get_unchecked_mut(index as usize) *= kda_decay_exp(value_g);
        *kneg.get_unchecked_mut(index as usize) = k[index as usize] * kda_decay_exp(-value_g);
    }
}

pub(super) fn make_kg_kpos_vbeta_body(
    mut k: DisjointSlice<f32>,
    mut v: DisjointSlice<f32>,
    g: &[f32],
    beta: &[f32],
    mut kpos_beta: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let Some(index) = thread_index(compact_elems(&params)) else {
        return;
    };
    let (dim, token, _bh, batch, head) = compact_linear_parts(index, &params);
    let chunk_end = chunk_end_token(token / params.chunk_size, &params);
    let g_value = g[index as usize];
    let g_last = g[compact_index(batch, chunk_end, head, dim, &params)];
    let beta_value = beta[beta_compact_index(batch, token, head, &params)];

    unsafe {
        let k_value = *k.get_unchecked_mut(index as usize);
        *k.get_unchecked_mut(index as usize) = k_value * kda_decay_exp(g_last - g_value);
        *v.get_unchecked_mut(index as usize) *= beta_value;
        *kpos_beta.get_unchecked_mut(index as usize) =
            k_value * beta_value * kda_decay_exp(g_value);
    }
}

pub(super) fn store_chunk_g_last_body(
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

pub(super) fn make_kneg_from_kg_body(
    k: &[f32],
    chunk_g_last: &[f32],
    mut kneg: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let Some(index) = thread_index(compact_elems(&params)) else {
        return;
    };
    let (dim, token, bh, _batch, _head) = compact_linear_parts(index, &params);
    let chunk = token / params.chunk_size;
    let g_last = chunk_g_last[chunk_g_last_index(bh, chunk, dim, &params)];
    unsafe {
        *kneg.get_unchecked_mut(index as usize) = k[index as usize] * kda_decay_exp(-g_last);
    }
}

pub(super) fn mask_aqk_body(mut aqk: DisjointSlice<f32>, params: CausalAttentionParams) {
    mask_chunk_matrix(&mut aqk, params, false);
}

pub(super) fn mask_akk_body(mut akk: DisjointSlice<f32>, params: CausalAttentionParams) {
    mask_chunk_matrix(&mut akk, params, true);
}

pub(super) fn solve_akk_inv_body(
    mut akk: DisjointSlice<f32>,
    params: CausalAttentionParams,
    raw: &mut SharedArray<f32, KDA_MATRIX_ELEMS>,
    inv: &mut SharedArray<f32, KDA_MATRIX_ELEMS>,
) {
    let batch_chunk = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    let chunks = chunk_count(&params);
    let matrix_elems = chunk_matrix_elems(&params);
    if batch_chunk >= batch_head(&params) * chunks || params.chunk_size > KDA_MAX_HEAD_DIM as u32 {
        return;
    }

    let base = (batch_chunk * matrix_elems) as usize;
    let mut idx = tid;
    while idx < matrix_elems {
        raw[idx as usize] = unsafe { *akk.get_unchecked_mut(base + idx as usize) };
        inv[idx as usize] = 0.0;
        idx += TC_FORWARD_THREADS_PER_BLOCK;
    }
    thread::sync_threads();

    let mut row = 0;
    while row < params.chunk_size {
        idx = tid;
        while idx <= row {
            let value = if idx == row {
                1.0
            } else {
                let mut sum = 0.0;
                let mut mid = idx;
                while mid < row {
                    sum = fma_f32(
                        raw[(row * params.chunk_size + mid) as usize],
                        inv[(mid * params.chunk_size + idx) as usize],
                        sum,
                    );
                    mid += 1;
                }
                -sum
            };
            inv[(row * params.chunk_size + idx) as usize] = value;
            idx += TC_FORWARD_THREADS_PER_BLOCK;
        }
        thread::sync_threads();
        row += 1;
    }

    idx = tid;
    while idx < matrix_elems {
        let row = idx / params.chunk_size;
        let col = idx - row * params.chunk_size;
        let value = if col <= row { inv[idx as usize] } else { 0.0 };
        unsafe {
            *akk.get_unchecked_mut(base + idx as usize) = value;
        }
        idx += TC_FORWARD_THREADS_PER_BLOCK;
    }
}

fn mask_chunk_matrix(matrix: &mut DisjointSlice<f32>, params: CausalAttentionParams, strict: bool) {
    let chunks = chunk_count(&params);
    let matrix_elems = chunk_matrix_elems(&params);
    let total = batch_head(&params) * chunks * matrix_elems;
    let Some(index) = thread_index(total) else {
        return;
    };
    let elem = index % matrix_elems;
    let row = elem / params.chunk_size;
    let col = elem - row * params.chunk_size;
    let keep = if strict { row > col } else { row >= col };
    if !keep {
        unsafe {
            *matrix.get_unchecked_mut(index as usize) = 0.0;
        }
    }
}

#[inline(always)]
fn thread_index(element_count: u32) -> Option<u32> {
    linear_thread_index(TC_FORWARD_THREADS_PER_BLOCK, element_count)
}
