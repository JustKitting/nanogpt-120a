use cuda_device::{DisjointSlice, thread};

use super::super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::cta_tile::CtaTile;
use crate::kda_backward::{
    load_chunk_state, stage_compact_t_a, stage_compact_t_a_disjoint, stage_hidden_dout_b_t,
    store_dh_quads,
};
use crate::kda_common::{batch_head, chunk_count, kda_tc_shape, state_elems};
use crate::kda_tc::{CompactStore::Add, CompactTileCtx, CtaATile, CtaBTile, KdaStateTile, stage_compact_a as stage_dm_compact_a, stage_compact_b_t_disjoint, stage_shared_state_b_t, store_compact_quads, tc_stage_loop};

#[derive(Clone, Copy)]
pub(crate) struct KdaChunkwiseInputs<'a> { pub(crate) qg: &'a [f32], pub(crate) kg: &'a [f32], pub(crate) g: &'a [f32], pub(crate) chunk_states: &'a [u16], pub(crate) d_out: &'a [f32] }
pub(crate) struct KdaChunkwiseGrads<'a> { pub(crate) u_to_du: DisjointSlice<'a, f32>, pub(crate) w_to_dw: DisjointSlice<'a, f32>, pub(crate) d_h_states: DisjointSlice<'a, f32> }
pub(crate) fn chunkwise_kda_backward_body(
    inputs: KdaChunkwiseInputs<'_>,
    mut grads: KdaChunkwiseGrads<'_>,
    params: CausalAttentionParams,
    states: (&mut KdaStateTile, &mut KdaStateTile, &mut KdaStateTile),
    tiles: (&mut CtaATile, &mut CtaBTile),
) {
    let (state, d_h_next, d_h) = states;
    let (a_tile, b_tile) = tiles;
    let bh = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    if bh >= batch_head(&params) || !kda_tc_shape(&params) {
        return;
    }

    let state_elems = state_elems(&params);
    let chunks = chunk_count(&params);
    let batch = bh / params.head_count;
    let head = bh - batch * params.head_count;

    let mut idx = tid;
    while idx < state_elems {
        d_h_next[idx as usize] = 0.0;
        idx += TC_BACKWARD_THREADS_PER_BLOCK;
    }
    thread::sync_threads();

    let mut chunk_remaining = chunks;
    while chunk_remaining > 0 {
        let chunk = chunk_remaining - 1;
        let start = chunk * params.chunk_size;
        let end = params.seq_len.min(start + params.chunk_size);
        load_chunk_state(
            inputs.chunk_states,
            state,
            bh,
            chunk,
            state_elems,
            &params,
            TC_BACKWARD_THREADS_PER_BLOCK,
        );

        idx = tid;
        let d_h_base = ((bh * chunks + chunk) * state_elems) as usize;
        while idx < state_elems {
            unsafe {
                *grads.d_h_states.get_unchecked_mut(d_h_base + idx as usize) = d_h_next[idx as usize];
            }
            idx += TC_BACKWARD_THREADS_PER_BLOCK;
        }

        let tile = CtaTile::from_tile(tid, 0, 0, 0);
        let ctx = CompactTileCtx::new(tile, (batch, head), (start, end), &params);
        add_kg_dh_to_du_tc(inputs.kg, &mut grads.u_to_du, d_h_next, a_tile, b_tile, ctx);
        thread::sync_threads();

        compute_prev_dh_tc(inputs, &mut grads, (&*d_h_next, &mut *d_h), (&mut *a_tile, &mut *b_tile), ctx);
        thread::sync_threads();

        idx = tid;
        while idx < state_elems {
            d_h_next[idx as usize] = d_h[idx as usize];
            idx += TC_BACKWARD_THREADS_PER_BLOCK;
        }
        thread::sync_threads();

        chunk_remaining -= 1;
    }
}

fn add_kg_dh_to_du_tc(
    kg: &[f32], d_u: &mut DisjointSlice<f32>,
    d_h_next: &KdaStateTile,
    a_tile: &mut CtaATile, b_tile: &mut CtaBTile,
    ctx: CompactTileCtx<'_>,
) {
    let mut acc = [[0.0_f32; 4]; 4];
    tc_stage_loop!(ctx.tile, a_tile, b_tile, acc; k_base < ctx.params.head_dim; {
        stage_dm_compact_a(kg, a_tile, ctx, k_base);
    } {
        stage_shared_state_b_t(d_h_next, b_tile, ctx, k_base);
    });

    store_compact_quads(acc, d_u, ctx, Add);
}

fn compute_prev_dh_tc(
    inputs: KdaChunkwiseInputs<'_>,
    grads: &mut KdaChunkwiseGrads<'_>,
    states: (&KdaStateTile, &mut KdaStateTile),
    tiles: (&mut CtaATile, &mut CtaBTile),
    compact_ctx: CompactTileCtx<'_>,
) {
    let (d_h_next, d_h) = states;
    let (a_tile, b_tile) = tiles;
    let mut acc = [[0.0_f32; 4]; 4];

    tc_stage_loop!(compact_ctx.tile, a_tile, b_tile, acc; k_base < compact_ctx.params.chunk_size; {
        stage_compact_t_a(inputs.qg, a_tile, compact_ctx, k_base, 1.0);
    } {
        stage_hidden_dout_b_t(inputs.d_out, b_tile, compact_ctx, k_base);
    });

    tc_stage_loop!(compact_ctx.tile, a_tile, b_tile, acc; k_base < compact_ctx.params.chunk_size; {
        stage_compact_t_a_disjoint(&mut grads.w_to_dw, a_tile, compact_ctx, k_base, -1.0);
    } {
        stage_compact_b_t_disjoint(&mut grads.u_to_du, b_tile, compact_ctx, k_base);
    });

    store_dh_quads(acc, d_h_next, d_h, inputs.g, compact_ctx);
}
