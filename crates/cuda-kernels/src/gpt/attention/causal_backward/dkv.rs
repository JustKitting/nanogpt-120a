use cuda_device::{DisjointSlice, SharedArray, thread, warp};

use super::dkv_accumulate::accumulate_key;
use super::dkv_thread::KeyThread;
use super::layout::qkv_index;
use super::reductions::KEY_REDUCE_PAIR_LEN;
use super::rope::rope_raw_grad;
use super::types::{
    CAUSAL_BACKWARD_HEAD_DIM_THREADS, CAUSAL_BACKWARD_KEY_BLOCK, CausalAttentionBackwardParams,
};

pub(super) fn dkv_body(
    qkv: &[f32],
    d_out: &[f32],
    log_sum_exp: &[f32],
    softmax_d: &[f32],
    mut d_qkv: DisjointSlice<f32>,
    params: CausalAttentionBackwardParams,
    reduce: &mut SharedArray<f32, KEY_REDUCE_PAIR_LEN>,
    prob: &mut SharedArray<f32, { CAUSAL_BACKWARD_KEY_BLOCK as usize }>,
    ds: &mut SharedArray<f32, { CAUSAL_BACKWARD_KEY_BLOCK as usize }>,
    dk_rot_shared: &mut SharedArray<
        f32,
        { (CAUSAL_BACKWARD_KEY_BLOCK * CAUSAL_BACKWARD_HEAD_DIM_THREADS) as usize },
    >,
) {
    let tid = thread::threadIdx_x();
    let head = thread::blockIdx_y();
    let state = KeyThread {
        key_offset: tid / CAUSAL_BACKWARD_HEAD_DIM_THREADS,
        dim: tid % CAUSAL_BACKWARD_HEAD_DIM_THREADS,
        lane: warp::lane_id(),
        warp_in_key: (tid % CAUSAL_BACKWARD_HEAD_DIM_THREADS) / 32,
        batch: thread::blockIdx_z(),
        head,
        block_key: thread::blockIdx_x() * CAUSAL_BACKWARD_KEY_BLOCK,
    };
    let valid = state.valid(&params);

    let (dk_rot, dv) = accumulate_key(
        qkv,
        d_out,
        log_sum_exp,
        softmax_d,
        &params,
        state,
        reduce,
        prob,
        ds,
    );

    dk_rot_shared[tid as usize] = dk_rot;
    thread::sync_threads();

    if valid {
        let pair = state.dim ^ 1;
        let paired =
            dk_rot_shared[(state.key_offset * CAUSAL_BACKWARD_HEAD_DIM_THREADS + pair) as usize];
        let key = state.key();
        let dk = rope_raw_grad(key, state.dim, dk_rot, paired, params.head_dim);
        unsafe {
            *d_qkv.get_unchecked_mut(qkv_index(
                state.batch,
                key,
                state.head,
                state.dim,
                params.embedding_dim,
                &params,
            )) = dk;
            *d_qkv.get_unchecked_mut(qkv_index(
                state.batch,
                key,
                state.head,
                state.dim,
                params.embedding_dim * 2,
                &params,
            )) = dv;
        }
    }
}
