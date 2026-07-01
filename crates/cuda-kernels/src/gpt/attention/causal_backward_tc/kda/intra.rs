use cuda_device::{DisjointSlice, thread};

mod beta;
mod compact;
mod prefix;

use beta::update_beta_grad;
use compact::update_compact_grads;
use prefix::reverse_prefix_dg;

use crate::attention::CausalAttentionParams;
use crate::kda_common::{beta_compact_index, compact_index};
use crate::kda_tc::KdaChunkTileCtx;

#[derive(Clone, Copy)]
pub(crate) struct KdaIntraInputs<'a> {
    pub(crate) qg: &'a [f32],
    pub(crate) kg: &'a [f32],
    pub(crate) vbeta: &'a [f32],
    pub(crate) g: &'a [f32],
    pub(crate) beta: &'a [f32],
    pub(crate) d_kg: &'a [f32],
    pub(crate) d_kpos_m: &'a [f32],
    pub(crate) d_vbeta_m: &'a [f32],
    pub(crate) d_kneg_from_b: &'a [f32],
    pub(crate) d_kpos_from_b_t: &'a [f32],
}

pub(crate) struct KdaIntraGrads<'a> {
    pub(crate) qg_to_dv: DisjointSlice<'a, f32>,
    pub(crate) k_a_to_dg: DisjointSlice<'a, f32>,
    pub(crate) q: DisjointSlice<'a, f32>,
    pub(crate) k: DisjointSlice<'a, f32>,
    pub(crate) beta: DisjointSlice<'a, f32>,
}

#[derive(Clone, Copy)]
struct KdaIntraCtx<'a> {
    params: &'a CausalAttentionParams,
    batch: u32,
    head: u32,
    start: u32,
    end: u32,
    head_dim: u32,
    chunk_tokens: u32,
}

impl KdaIntraCtx<'_> {
    fn compact(self, token: u32, dim: u32) -> usize {
        compact_index(self.batch, token, self.head, dim, self.params)
    }

    fn beta(self, token: u32) -> usize {
        beta_compact_index(self.batch, token, self.head, self.params)
    }

    fn last_compact(self, dim: u32) -> usize {
        self.compact(self.end - 1, dim)
    }
}

pub(crate) fn chunk_intra_kda_backward_body(
    inputs: KdaIntraInputs<'_>,
    mut grads: KdaIntraGrads<'_>,
    params: CausalAttentionParams,
) {
    let Some(ctx) = KdaChunkTileCtx::from_block(&params) else {
        return;
    };
    let tid = thread::threadIdx_x();
    let compact_ctx = ctx.compact;
    let ctx = KdaIntraCtx {
        params: &params,
        batch: compact_ctx.batch,
        head: compact_ctx.head,
        start: compact_ctx.start,
        end: compact_ctx.end,
        head_dim: params.head_dim,
        chunk_tokens: ctx.matrix.chunk_tokens,
    };

    update_compact_grads(inputs, &mut grads, ctx, tid);
    thread::sync_threads();

    update_beta_grad(inputs, &mut grads.beta, ctx, tid);
    thread::sync_threads();

    reverse_prefix_dg(&mut grads.k_a_to_dg, ctx, tid);
}
