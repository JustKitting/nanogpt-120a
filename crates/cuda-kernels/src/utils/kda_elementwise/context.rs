use cuda_device::{thread, warp};

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::cvt_f32_f16;
use crate::float_ptx::fma_f32;
use crate::kda_common::{KDA_MAX_HEAD_DIM, k_offset, kda_warp_norm, q_offset, qkv_index, silu};

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

#[derive(Clone, Copy)]
pub(crate) struct KdaQkNormAcc {
    q_sum: f32,
    k_sum: f32,
}

impl KdaQkNormAcc {
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        Self {
            q_sum: 0.0,
            k_sum: 0.0,
        }
    }

    #[inline(always)]
    pub(crate) fn add(&mut self, qk: KdaQkAct) {
        self.q_sum = fma_f32(qk.q_act, qk.q_act, self.q_sum);
        self.k_sum = fma_f32(qk.k_act, qk.k_act, self.k_sum);
    }

    #[inline(always)]
    pub(crate) fn norms(self) -> (f32, f32) {
        (kda_warp_norm(self.q_sum), kda_warp_norm(self.k_sum))
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
