use cuda_device::{DisjointSlice, thread, warp};

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::cvt_f32_f16;
use crate::float_ptx::{fma_f32, sqrt_f32};
use crate::kda_common::{
    KDA_DENOM_EPS, KDA_MAX_HEAD_DIM, batch_head, beta_compact_index, beta_index, compact_index,
    g_offset, k_offset, q_offset, qkv_index, safe_denom, sigmoid, silu, softplus, v_offset,
};
use crate::warp_reduce::warp_sum_f32;

pub(crate) trait KdaQkvRead: Sized {
    fn read(values: &[Self], index: usize) -> f32;
}

impl KdaQkvRead for f32 {
    #[inline(always)]
    fn read(values: &[Self], index: usize) -> f32 {
        values[index]
    }
}

impl KdaQkvRead for u16 {
    #[inline(always)]
    fn read(values: &[Self], index: usize) -> f32 {
        cvt_f32_f16(values[index])
    }
}

#[derive(Clone, Copy)]
pub(crate) struct KdaWarpCtx {
    pub(crate) lane: u32,
    pub(crate) row: u32,
    pub(crate) batch: u32,
    pub(crate) token: u32,
    pub(crate) head: u32,
    pub(crate) valid: bool,
}

pub(crate) fn kda_warp_ctx(threads_per_block: u32, params: &CausalAttentionParams) -> KdaWarpCtx {
    let thread_index = thread::blockIdx_x() * threads_per_block + thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_in_grid = thread_index / 32;
    let row = warp_in_grid / params.head_count;
    let head = warp_in_grid - row * params.head_count;
    KdaWarpCtx {
        lane,
        row,
        batch: row / params.seq_len,
        token: row % params.seq_len,
        head,
        valid: row < params.row_count && params.head_dim <= KDA_MAX_HEAD_DIM as u32,
    }
}

#[derive(Clone, Copy)]
pub(crate) struct KdaQkAct {
    pub(crate) raw_q: f32,
    pub(crate) raw_k: f32,
    pub(crate) q_act: f32,
    pub(crate) k_act: f32,
}

impl KdaQkAct {
    pub(crate) const fn zero() -> Self {
        Self {
            raw_q: 0.0,
            raw_k: 0.0,
            q_act: 0.0,
            k_act: 0.0,
        }
    }
}

pub(crate) fn read_qk_act<T: KdaQkvRead>(
    qkv: &[T],
    row: u32,
    head: u32,
    dim: u32,
    params: &CausalAttentionParams,
) -> KdaQkAct {
    let raw_q = T::read(qkv, qkv_index(row, head, dim, q_offset(params), params));
    let raw_k = T::read(qkv, qkv_index(row, head, dim, k_offset(params), params));
    KdaQkAct {
        raw_q,
        raw_k,
        q_act: silu(raw_q),
        k_act: silu(raw_k),
    }
}

pub(crate) fn chunk_cumsum_g_body(mut g: DisjointSlice<f32>, params: CausalAttentionParams) {
    let bh = thread::blockIdx_x();
    let chunk = thread::blockIdx_y();
    let dim = thread::threadIdx_x();
    if bh >= batch_head(&params)
        || chunk * params.chunk_size >= params.seq_len
        || dim >= params.head_dim
    {
        return;
    }

    let batch = bh / params.head_count;
    let head = bh - batch * params.head_count;
    let start = chunk * params.chunk_size;
    let end = params.seq_len.min(start + params.chunk_size);
    let mut acc = 0.0;
    let mut token = start;
    while token < end {
        let compact = compact_index(batch, token, head, dim, &params);
        unsafe {
            acc += *g.get_unchecked_mut(compact);
            *g.get_unchecked_mut(compact) = acc;
        }
        token += 1;
    }
}

pub(crate) fn prepare_kda_inputs_body<T: KdaQkvRead>(
    qkv: &[T],
    mut q: DisjointSlice<f32>,
    mut k: DisjointSlice<f32>,
    mut v: DisjointSlice<f32>,
    mut g: DisjointSlice<f32>,
    mut beta: DisjointSlice<f32>,
    params: CausalAttentionParams,
    threads_per_block: u32,
) {
    let ctx = kda_warp_ctx(threads_per_block, &params);
    if !ctx.valid {
        return;
    }

    let mut q_sum = 0.0;
    let mut k_sum = 0.0;
    let mut qk0 = KdaQkAct::zero();
    let dim0 = ctx.lane;
    if dim0 < params.head_dim {
        qk0 = read_qk_act(qkv, ctx.row, ctx.head, dim0, &params);
        q_sum = fma_f32(qk0.q_act, qk0.q_act, q_sum);
        k_sum = fma_f32(qk0.k_act, qk0.k_act, k_sum);
    }

    let mut qk1 = KdaQkAct::zero();
    let dim1 = ctx.lane + 32;
    if dim1 < params.head_dim {
        qk1 = read_qk_act(qkv, ctx.row, ctx.head, dim1, &params);
        q_sum = fma_f32(qk1.q_act, qk1.q_act, q_sum);
        k_sum = fma_f32(qk1.k_act, qk1.k_act, k_sum);
    }

    let q_norm = sqrt_f32(warp_sum_f32(q_sum) + KDA_DENOM_EPS);
    let k_norm = sqrt_f32(warp_sum_f32(k_sum) + KDA_DENOM_EPS);
    let q_inv = params.scale / safe_denom(q_norm);
    let k_inv = 1.0 / safe_denom(k_norm);

    if dim0 < params.head_dim {
        write_prepared(
            qkv, &mut q, &mut k, &mut v, &mut g, ctx, dim0, qk0, q_inv, k_inv, &params,
        );
    }
    if dim1 < params.head_dim {
        write_prepared(
            qkv, &mut q, &mut k, &mut v, &mut g, ctx, dim1, qk1, q_inv, k_inv, &params,
        );
    }
    if ctx.lane == 0 {
        let raw_beta = T::read(qkv, beta_index(ctx.row, ctx.head, &params));
        unsafe {
            *beta.get_unchecked_mut(beta_compact_index(ctx.batch, ctx.token, ctx.head, &params)) =
                sigmoid(raw_beta);
        }
    }
}

fn write_prepared<T: KdaQkvRead>(
    qkv: &[T],
    q: &mut DisjointSlice<f32>,
    k: &mut DisjointSlice<f32>,
    v: &mut DisjointSlice<f32>,
    g: &mut DisjointSlice<f32>,
    ctx: KdaWarpCtx,
    dim: u32,
    qk: KdaQkAct,
    q_inv: f32,
    k_inv: f32,
    params: &CausalAttentionParams,
) {
    let compact = compact_index(ctx.batch, ctx.token, ctx.head, dim, params);
    let raw_v = T::read(
        qkv,
        qkv_index(ctx.row, ctx.head, dim, v_offset(params), params),
    );
    let raw_g = T::read(
        qkv,
        qkv_index(ctx.row, ctx.head, dim, g_offset(params), params),
    );
    unsafe {
        *q.get_unchecked_mut(compact) = qk.q_act * q_inv;
        *k.get_unchecked_mut(compact) = qk.k_act * k_inv;
        *v.get_unchecked_mut(compact) = silu(raw_v);
        *g.get_unchecked_mut(compact) = -params.decay_scale * softplus(raw_g);
    }
}
