use cuda_device::{DisjointSlice, thread};

use crate::attention::CausalAttentionParams;
use crate::kda_common::{
    batch_head, beta_compact_index, beta_index, compact_index, g_offset, qkv_index, safe_denom,
    sigmoid, silu, softplus, v_offset,
};

use super::context::{KdaQkAct, KdaQkNormAcc, KdaQkvRead, KdaWarpCtx, kda_warp_ctx, read_qk_act};

pub(crate) struct KdaPrepareOutputs<'a> {
    pub(crate) q: DisjointSlice<'a, f32>,
    pub(crate) k: DisjointSlice<'a, f32>,
    pub(crate) v: DisjointSlice<'a, f32>,
    pub(crate) g: DisjointSlice<'a, f32>,
    pub(crate) beta: DisjointSlice<'a, f32>,
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
    mut out: KdaPrepareOutputs<'_>,
    params: CausalAttentionParams,
    threads_per_block: u32,
) {
    let ctx = kda_warp_ctx(threads_per_block, &params);
    if !ctx.valid {
        return;
    }

    let mut acc = KdaQkNormAcc::zero();
    let dim0 = ctx.lane;
    let qk0 = read_prepare_dim(qkv, ctx, dim0, &params, &mut acc);
    let dim1 = ctx.lane + 32;
    let qk1 = read_prepare_dim(qkv, ctx, dim1, &params, &mut acc);
    let (q_norm, k_norm) = acc.norms();
    let inv = (params.scale / safe_denom(q_norm), 1.0 / safe_denom(k_norm));

    if dim0 < params.head_dim {
        write_prepared(qkv, &mut out, ctx, dim0, qk0, inv, &params);
    }
    if dim1 < params.head_dim {
        write_prepared(qkv, &mut out, ctx, dim1, qk1, inv, &params);
    }
    if ctx.lane == 0 {
        let raw_beta = T::read(qkv, beta_index(ctx.row, ctx.head, &params));
        unsafe {
            *out.beta
                .get_unchecked_mut(beta_compact_index(ctx.batch, ctx.token, ctx.head, &params)) =
                sigmoid(raw_beta);
        }
    }
}

#[inline(always)]
fn read_prepare_dim<T: KdaQkvRead>(
    qkv: &[T],
    ctx: KdaWarpCtx,
    dim: u32,
    params: &CausalAttentionParams,
    acc: &mut KdaQkNormAcc,
) -> KdaQkAct {
    if dim >= params.head_dim {
        return KdaQkAct::zero();
    }

    let qk = read_qk_act(qkv, ctx.row, ctx.head, dim, params);
    acc.add(qk);
    qk
}

fn write_prepared<T: KdaQkvRead>(
    qkv: &[T],
    out: &mut KdaPrepareOutputs<'_>,
    ctx: KdaWarpCtx,
    dim: u32,
    qk: KdaQkAct,
    inv: (f32, f32),
    params: &CausalAttentionParams,
) {
    let (q_inv, k_inv) = inv;
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
        *out.q.get_unchecked_mut(compact) = qk.q_act * q_inv;
        *out.k.get_unchecked_mut(compact) = qk.k_act * k_inv;
        *out.v.get_unchecked_mut(compact) = silu(raw_v);
        *out.g.get_unchecked_mut(compact) = -params.decay_scale * softplus(raw_g);
    }
}
