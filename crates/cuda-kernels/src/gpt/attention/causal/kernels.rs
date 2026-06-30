use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use super::{CAUSAL_MAX_WARPS_PER_BLOCK, CausalAttentionParams};
use crate::attention::layout::batched_qkv_index;
use crate::block_reduce::block_reduce_f32;
use crate::float_ptx::{exp_f32, fma_f32, ln_f32, max_f32, safe_positive_denom};
use crate::warp_reduce::{thread_lane_warp, warp_max_f32, warp_sum_f32};

const MAX_CAUSAL_TOKENS: usize = 1024;
const NEG_INFINITY: f32 = -3.4028235e38_f32;

pub use module::{LoadedModule, from_module};

#[cuda_module]
pub mod module {
    use super::*;

    static mut SCORES: SharedArray<f32, MAX_CAUSAL_TOKENS> = SharedArray::UNINIT;
    static mut REDUCE: SharedArray<f32, { CAUSAL_MAX_WARPS_PER_BLOCK as usize }> =
        SharedArray::UNINIT;

    macro_rules! score_reduce {
        ($query:expr, $thread_index:expr, $init:expr, $warp_reduce:ident, |$local:ident, $key:ident| $value:expr) => {{
            let lane = warp::lane_id();
            let warp_in_block = $thread_index / 32;
            let block_threads = thread::blockDim_x();
            let mut $local = $init;
            let mut $key = $thread_index;

            while $key <= $query {
                unsafe {
                    $local = $value;
                }
                $key += block_threads;
            }
            block_reduce_f32!(
                REDUCE,
                block_threads / 32,
                $local,
                lane,
                warp_in_block,
                $warp_reduce,
                $init
            )
        }};
    }

    #[kernel]
    pub fn causal_attention_kernel(
        qkv: &[f32],
        mut out: DisjointSlice<f32>,
        mut log_sum_exp: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        let query = thread::blockIdx_x();
        let head = thread::blockIdx_y();
        let batch = thread::blockIdx_z();
        let (thread_index, lane, warp_in_block) = thread_lane_warp();

        let row = batch * params.seq_len + query;
        if query >= params.seq_len
            || head >= params.head_count
            || batch >= params.batch_size
            || row >= params.row_count
        {
            return;
        }

        let query_value = if thread_index < params.head_dim {
            qkv_value(qkv, batch, query, head, thread_index, 0, &params)
        } else {
            0.0
        };
        let mut key = 0;
        while key <= query {
            let local_dot = if thread_index < params.head_dim {
                query_value
                    * qkv_value(
                        qkv,
                        batch,
                        key,
                        head,
                        thread_index,
                        params.embedding_dim,
                        &params,
                    )
            } else {
                0.0
            };
            let dot = block_reduce_f32!(
                REDUCE,
                thread::blockDim_x() / 32,
                local_dot,
                lane,
                warp_in_block,
                warp_sum_f32,
                0.0
            );
            if thread_index == 0 {
                unsafe {
                    SCORES[key as usize] = dot * params.scale;
                }
            }

            thread::sync_threads();
            key += 1;
        }

        let score_max = score_max(query, thread_index);
        let denom = safe_positive_denom(score_denom(query, thread_index, score_max));
        if thread_index == 0 {
            let log_sum_exp_index = (batch as usize * params.head_count as usize + head as usize)
                * params.seq_len as usize
                + query as usize;
            unsafe {
                *log_sum_exp.get_unchecked_mut(log_sum_exp_index) = score_max + ln_f32(denom);
            }
        }

        if thread_index < params.head_dim {
            let mut value = 0.0;
            key = 0;
            while key <= query {
                let score = unsafe { SCORES[key as usize] };
                let weight = exp_f32(score - score_max) / denom;
                value = fma_f32(
                    weight,
                    qkv_value(
                        qkv,
                        batch,
                        key,
                        head,
                        thread_index,
                        params.embedding_dim * 2,
                        &params,
                    ),
                    value,
                );
                key += 1;
            }

            let out_index = row as usize * params.embedding_dim as usize
                + head as usize * params.head_dim as usize
                + thread_index as usize;
            unsafe {
                *out.get_unchecked_mut(out_index) = value;
            }
        }
    }

    #[inline(always)]
    fn score_max(query: u32, thread_index: u32) -> f32 {
        score_reduce!(
            query,
            thread_index,
            NEG_INFINITY,
            warp_max_f32,
            |local, key| max_f32(local, SCORES[key as usize])
        )
    }

    #[inline(always)]
    fn score_denom(query: u32, thread_index: u32, score_max: f32) -> f32 {
        score_reduce!(query, thread_index, 0.0, warp_sum_f32, |local, key| local
            + exp_f32(SCORES[key as usize] - score_max))
    }

    #[inline(always)]
    fn qkv_value(
        qkv: &[f32],
        batch: u32,
        token: u32,
        head: u32,
        dim: u32,
        section_offset: u32,
        params: &CausalAttentionParams,
    ) -> f32 {
        qkv[batched_qkv_index(batch, token, head, dim, section_offset, params)]
    }
}
