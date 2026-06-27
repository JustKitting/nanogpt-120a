use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use super::AttentionModule;
use crate::float_ptx::{exp_f32, fma_f32, ln_f32, max_f32, safe_positive_denom};
use crate::warp_reduce::{warp_max_f32, warp_sum_f32};

pub(crate) const CAUSAL_ATTENTION_MAX_THREADS_PER_BLOCK: u32 = 128;
pub(crate) const CAUSAL_MAX_WARPS_PER_BLOCK: u32 = CAUSAL_ATTENTION_MAX_THREADS_PER_BLOCK / 32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CausalAttentionParams {
    pub row_count: u32,
    pub seq_len: u32,
    pub batch_size: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub chunk_size: u32,
    pub decay_scale: f32,
}

unsafe impl DeviceCopy for CausalAttentionParams {}

pub struct CausalAttentionArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub qkv: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub log_sum_exp: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub seq_len: u32,
    pub batch_size: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
}

impl AttentionModule {
    pub fn causal_attention(&self, args: CausalAttentionArgs<'_, '_>) -> Result<(), DriverError> {
        self.causal_attention.causal_attention_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.seq_len, args.head_count, args.batch_size),
                block_dim: (causal_attention_threads(args.head_dim), 1, 1),
                shared_mem_bytes: 0,
            },
            args.qkv,
            args.out,
            args.log_sum_exp,
            CausalAttentionParams {
                row_count: args.row_count,
                seq_len: args.seq_len,
                batch_size: args.batch_size,
                embedding_dim: args.embedding_dim,
                qkv_dim: args.qkv_dim,
                head_count: args.head_count,
                head_dim: args.head_dim,
                scale: 1.0 / (args.head_dim as f32).sqrt(),
                chunk_size: 64,
                decay_scale: 0.01,
            },
        )
    }
}

fn causal_attention_threads(head_dim: u32) -> u32 {
    let threads = head_dim.div_ceil(32) * 32;
    assert!(threads <= CAUSAL_ATTENTION_MAX_THREADS_PER_BLOCK);
    threads.max(32)
}

#[allow(static_mut_refs)]
#[cuda_module]
pub mod kernels {
    use super::*;

    const MAX_CAUSAL_TOKENS: usize = 1024;
    const NEG_INFINITY: f32 = -3.4028235e38_f32;

    static mut SCORES: SharedArray<f32, MAX_CAUSAL_TOKENS> = SharedArray::UNINIT;
    static mut REDUCE: SharedArray<f32, { CAUSAL_MAX_WARPS_PER_BLOCK as usize }> =
        SharedArray::UNINIT;

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
        let thread_index = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = thread_index / 32;

        let row = batch * params.seq_len + query;
        if query >= params.seq_len
            || head >= params.head_count
            || batch >= params.batch_size
            || row >= params.row_count
        {
            return;
        }

        let query_value = if thread_index < params.head_dim {
            q_value(qkv, batch, query, head, thread_index, &params)
        } else {
            0.0
        };
        let mut key = 0;
        while key <= query {
            let local_dot = if thread_index < params.head_dim {
                query_value * k_value(qkv, batch, key, head, thread_index, &params)
            } else {
                0.0
            };
            let warp_dot = warp_sum_f32(local_dot);

            if lane == 0 {
                unsafe {
                    REDUCE[warp_in_block as usize] = warp_dot;
                }
            }

            thread::sync_threads();

            if thread_index == 0 {
                let mut dot = 0.0;
                let mut warp_index = 0;
                while warp_index < thread::blockDim_x() / 32 {
                    unsafe {
                        dot += REDUCE[warp_index as usize];
                    }
                    warp_index += 1;
                }

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
                    v_value(qkv, batch, key, head, thread_index, &params),
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
        let lane = warp::lane_id();
        let warp_in_block = thread_index / 32;
        let block_threads = thread::blockDim_x();
        let warp_count = block_threads / 32;
        let mut local_max = NEG_INFINITY;
        let mut key = thread_index;

        while key <= query {
            unsafe {
                local_max = max_f32(local_max, SCORES[key as usize]);
            }
            key += block_threads;
        }
        let warp_max = warp_max_f32(local_max);

        if lane == 0 {
            unsafe {
                REDUCE[warp_in_block as usize] = warp_max;
            }
        }

        thread::sync_threads();

        if thread_index == 0 {
            let mut block_max = NEG_INFINITY;
            let mut warp_index = 0;
            while warp_index < warp_count {
                unsafe {
                    block_max = max_f32(block_max, REDUCE[warp_index as usize]);
                }
                warp_index += 1;
            }
            unsafe {
                REDUCE[0] = block_max;
            }
        }

        thread::sync_threads();

        unsafe { REDUCE[0] }
    }

    #[inline(always)]
    fn score_denom(query: u32, thread_index: u32, score_max: f32) -> f32 {
        let lane = warp::lane_id();
        let warp_in_block = thread_index / 32;
        let block_threads = thread::blockDim_x();
        let warp_count = block_threads / 32;
        let mut local_sum = 0.0;
        let mut key = thread_index;

        while key <= query {
            unsafe {
                local_sum += exp_f32(SCORES[key as usize] - score_max);
            }
            key += block_threads;
        }
        let warp_total = warp_sum_f32(local_sum);

        if lane == 0 {
            unsafe {
                REDUCE[warp_in_block as usize] = warp_total;
            }
        }

        thread::sync_threads();

        if thread_index == 0 {
            let mut denom = 0.0;
            let mut warp_index = 0;
            while warp_index < warp_count {
                unsafe {
                    denom += REDUCE[warp_index as usize];
                }
                warp_index += 1;
            }
            unsafe {
                REDUCE[0] = denom;
            }
        }

        thread::sync_threads();

        unsafe { REDUCE[0] }
    }

    #[inline(always)]
    fn q_value(
        qkv: &[f32],
        batch: u32,
        token: u32,
        head: u32,
        dim: u32,
        params: &CausalAttentionParams,
    ) -> f32 {
        qkv[qkv_index(batch, token, head, dim, 0, params)]
    }

    #[inline(always)]
    fn k_value(
        qkv: &[f32],
        batch: u32,
        token: u32,
        head: u32,
        dim: u32,
        params: &CausalAttentionParams,
    ) -> f32 {
        qkv[qkv_index(batch, token, head, dim, params.embedding_dim, params)]
    }

    #[inline(always)]
    fn v_value(
        qkv: &[f32],
        batch: u32,
        token: u32,
        head: u32,
        dim: u32,
        params: &CausalAttentionParams,
    ) -> f32 {
        qkv[qkv_index(batch, token, head, dim, params.embedding_dim * 2, params)]
    }

    #[inline(always)]
    fn qkv_index(
        batch: u32,
        token: u32,
        head: u32,
        dim: u32,
        section_offset: u32,
        params: &CausalAttentionParams,
    ) -> usize {
        (batch as usize * params.seq_len as usize + token as usize) * params.qkv_dim as usize
            + section_offset as usize
            + head as usize * params.head_dim as usize
            + dim as usize
    }
}
