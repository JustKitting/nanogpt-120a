use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::block_reduce::block_max_shared_f32;
use crate::f16_tc_matmul::convert::cvt_f32_f16;
use crate::float_ptx::{fma_f32, sqrt_f32};
use crate::kda_common::{KDA_DENOM_EPS, silu};
use crate::warp_reduce::thread_lane_warp;

use super::threads::{MATRIX_THREADS_PER_BLOCK, WARPS_PER_BLOCK};

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn kda_aurora_qk_clip_kernel(
        qkv: &[u16],
        mut z_master: DisjointSlice<f32>,
        mut x_master: DisjointSlice<f32>,
        mut momentum: DisjointSlice<f32>,
        mut scores: DisjointSlice<f32>,
        row_count: u32,
        qkv_dim: u32,
        input_dim: u32,
        embedding_dim: u32,
        head_count: u32,
        head_dim: u32,
        tau: f32,
        silu_qk: u32,
    ) {
        let head = thread::blockIdx_x();
        if head >= head_count {
            return;
        }

        static mut REDUCE: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;
        let (tid, lane, warp_id) = thread_lane_warp();
        let mut q_max = 0.0;
        let mut k_max = 0.0;
        let params = ClipParams {
            qkv_dim,
            embedding_dim,
            head_dim,
        };

        let mut row = tid;
        while row < row_count {
            let (q_norm, k_norm) = qk_norms(qkv, row, head, params, silu_qk);
            q_max = if q_norm > q_max { q_norm } else { q_max };
            k_max = if k_norm > k_max { k_norm } else { k_max };
            row += MATRIX_THREADS_PER_BLOCK;
        }

        let q_max = unsafe { block_max_shared_f32(&mut REDUCE, q_max, lane, warp_id) };
        let k_max = unsafe { block_max_shared_f32(&mut REDUCE, k_max, lane, warp_id) };
        let score = q_max * k_max / sqrt_f32(head_dim as f32);
        let factor = clip_factor(score, tau);
        if tid == 0 {
            unsafe {
                *scores.get_unchecked_mut(head as usize) = score;
            }
        }
        if factor == 1.0 {
            return;
        }

        let per_side = head_dim * input_dim;
        let total = per_side * 2;
        let mut index = tid;
        while index < total {
            let side = index / per_side;
            let rem = index - side * per_side;
            let dim = rem / input_dim;
            let input_row = rem - dim * input_dim;
            let offset = if side == 0 { 0 } else { embedding_dim };
            let col = offset + head * head_dim + dim;
            let master_index = (col * input_dim + input_row) as usize;
            unsafe {
                let z = z_master.get_unchecked_mut(master_index);
                let x = x_master.get_unchecked_mut(master_index);
                let m = momentum.get_unchecked_mut(master_index);
                *z *= factor;
                *x *= factor;
                *m *= factor;
            }
            index += MATRIX_THREADS_PER_BLOCK;
        }
    }
}

#[derive(Clone, Copy)]
struct ClipParams {
    qkv_dim: u32,
    embedding_dim: u32,
    head_dim: u32,
}

fn qk_norms(qkv: &[u16], row: u32, head: u32, params: ClipParams, silu_qk: u32) -> (f32, f32) {
    let mut q_sum = 0.0;
    let mut k_sum = 0.0;
    let mut dim = 0;
    while dim < params.head_dim {
        let raw_q = cvt_f32_f16(qkv[qkv_index(row, head, dim, 0, params)]);
        let raw_k = cvt_f32_f16(qkv[qkv_index(row, head, dim, params.embedding_dim, params)]);
        let q = if silu_qk != 0 { silu(raw_q) } else { raw_q };
        let k = if silu_qk != 0 { silu(raw_k) } else { raw_k };
        q_sum = fma_f32(q, q, q_sum);
        k_sum = fma_f32(k, k, k_sum);
        dim += 1;
    }
    (
        sqrt_f32(q_sum + KDA_DENOM_EPS),
        sqrt_f32(k_sum + KDA_DENOM_EPS),
    )
}

fn qkv_index(row: u32, head: u32, dim: u32, section_offset: u32, params: ClipParams) -> usize {
    (row * params.qkv_dim + section_offset + head * params.head_dim + dim) as usize
}

#[allow(clippy::eq_op)]
fn clip_factor(score: f32, tau: f32) -> f32 {
    if score == score && score > tau {
        sqrt_f32(tau / (score + KDA_DENOM_EPS))
    } else {
        1.0
    }
}

pub(super) const KDA_CLIP_THREADS_PER_BLOCK: u32 = MATRIX_THREADS_PER_BLOCK;
