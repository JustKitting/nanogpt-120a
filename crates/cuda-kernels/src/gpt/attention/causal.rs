use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use super::AttentionModule;
use crate::float_ptx::{exp_f32, fma_f32, ln_f32, max_f32, sincos_f32};
use crate::warp_reduce::{warp_max_f32, warp_sum_f32};

pub(crate) const CAUSAL_ATTENTION_THREADS_PER_BLOCK: u32 = 64;
pub(crate) const CAUSAL_WARPS_PER_BLOCK: u32 = CAUSAL_ATTENTION_THREADS_PER_BLOCK / 32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CausalAttentionParams {
    pub token_count: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
    pub scale: f32,
}

unsafe impl DeviceCopy for CausalAttentionParams {}

pub struct CausalAttentionArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub qkv: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub lse: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
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
                grid_dim: (args.token_count, args.head_count, 1),
                block_dim: (CAUSAL_ATTENTION_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.qkv,
            args.out,
            args.lse,
            CausalAttentionParams {
                token_count: args.token_count,
                embedding_dim: args.embedding_dim,
                qkv_dim: args.qkv_dim,
                head_count: args.head_count,
                head_dim: args.head_dim,
                scale: 1.0 / (args.head_dim as f32).sqrt(),
            },
        )
    }
}

#[allow(static_mut_refs)]
#[cuda_module]
pub mod kernels {
    use super::*;

    const MAX_CAUSAL_TOKENS: usize = 1024;
    const NEG_INFINITY: f32 = -3.4028235e38_f32;

    static mut SCORES: SharedArray<f32, MAX_CAUSAL_TOKENS> = SharedArray::UNINIT;
    static mut REDUCE: SharedArray<f32, { CAUSAL_WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

    #[kernel]
    pub fn causal_attention_kernel(
        qkv: &[f32],
        mut out: DisjointSlice<f32>,
        mut lse: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        let query = thread::blockIdx_x();
        let head = thread::blockIdx_y();
        let thread = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = thread / 32;

        if query >= params.token_count || head >= params.head_count {
            return;
        }

        let query_value = if thread < params.head_dim {
            rope_q_value(qkv, query, head, thread, &params)
        } else {
            0.0
        };
        let mut key = 0;
        while key <= query {
            let local_dot = if thread < params.head_dim {
                query_value * rope_k_value(qkv, key, head, thread, &params)
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

            if thread == 0 {
                let mut dot = 0.0;
                let mut warp_index = 0;
                while warp_index < CAUSAL_WARPS_PER_BLOCK {
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

        let score_max = score_max(query, thread);
        let denom = score_denom(query, thread, score_max);
        if thread == 0 {
            let lse_index = head as usize * params.token_count as usize + query as usize;
            unsafe {
                *lse.get_unchecked_mut(lse_index) = score_max + ln_f32(denom);
            }
        }

        if thread < params.head_dim {
            let mut value = 0.0;
            key = 0;
            while key <= query {
                let score = unsafe { SCORES[key as usize] };
                let weight = exp_f32(score - score_max) / denom;
                value = fma_f32(weight, v_value(qkv, key, head, thread, &params), value);
                key += 1;
            }

            let out_index = query as usize * params.embedding_dim as usize
                + head as usize * params.head_dim as usize
                + thread as usize;
            unsafe {
                *out.get_unchecked_mut(out_index) = value;
            }
        }
    }

    #[inline(always)]
    fn score_max(query: u32, thread_index: u32) -> f32 {
        let lane = warp::lane_id();
        let warp_in_block = thread_index / 32;
        let mut local_max = NEG_INFINITY;
        let mut key = thread_index;

        while key <= query {
            unsafe {
                local_max = max_f32(local_max, SCORES[key as usize]);
            }
            key += CAUSAL_ATTENTION_THREADS_PER_BLOCK;
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
            while warp_index < CAUSAL_WARPS_PER_BLOCK {
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
        let mut local_sum = 0.0;
        let mut key = thread_index;

        while key <= query {
            unsafe {
                local_sum += exp_f32(SCORES[key as usize] - score_max);
            }
            key += CAUSAL_ATTENTION_THREADS_PER_BLOCK;
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
            while warp_index < CAUSAL_WARPS_PER_BLOCK {
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
    fn rope_q_value(
        qkv: &[f32],
        token: u32,
        head: u32,
        dim: u32,
        params: &CausalAttentionParams,
    ) -> f32 {
        rope_value(qkv, token, head, dim, 0, params)
    }

    #[inline(always)]
    fn rope_k_value(
        qkv: &[f32],
        token: u32,
        head: u32,
        dim: u32,
        params: &CausalAttentionParams,
    ) -> f32 {
        rope_value(qkv, token, head, dim, params.embedding_dim, params)
    }

    #[inline(always)]
    fn rope_value(
        qkv: &[f32],
        token: u32,
        head: u32,
        dim: u32,
        section_offset: u32,
        params: &CausalAttentionParams,
    ) -> f32 {
        let paired_dim = if dim & 1 == 0 { dim + 1 } else { dim - 1 };
        let value = qkv[qkv_index(token, head, dim, section_offset, params)];
        let paired = qkv[qkv_index(token, head, paired_dim, section_offset, params)];
        let theta = token as f32 * rope_inv_freq(dim, params.head_dim);
        let (sin, cos) = sincos_f32(theta);

        if dim & 1 == 0 {
            fma_f32(-paired, sin, value * cos)
        } else {
            fma_f32(paired, sin, value * cos)
        }
    }

    #[inline(always)]
    fn rope_inv_freq(dim: u32, head_dim: u32) -> f32 {
        const ROPE_LN_BASE: f32 = 9.210_340_5;
        let pair_dim = (dim & !1) as f32;
        exp_f32(-ROPE_LN_BASE * pair_dim / head_dim as f32)
    }

    #[inline(always)]
    fn v_value(
        qkv: &[f32],
        token: u32,
        head: u32,
        dim: u32,
        params: &CausalAttentionParams,
    ) -> f32 {
        qkv[qkv_index(token, head, dim, params.embedding_dim * 2, params)]
    }

    #[inline(always)]
    fn qkv_index(
        token: u32,
        head: u32,
        dim: u32,
        section_offset: u32,
        params: &CausalAttentionParams,
    ) -> usize {
        token as usize * params.qkv_dim as usize
            + section_offset as usize
            + head as usize * params.head_dim as usize
            + dim as usize
    }
}
