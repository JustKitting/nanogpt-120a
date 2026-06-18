use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::float_ptx::sqrt_f32;

use super::super::threads::{MATRIX_THREADS_PER_BLOCK, WARP_SIZE, WARPS_PER_BLOCK};

#[allow(static_mut_refs)]
#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn row_inv_norm_kernel(
        x: &[f32],
        mut row_scale: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
        eps: f32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = tid / WARP_SIZE;

        if row < rows {
            let mut local = 0.0;
            let mut col = tid;
            while col < cols {
                let value = x[(row * cols + col) as usize];
                local += value * value;
                col += MATRIX_THREADS_PER_BLOCK;
            }

            let sum = block_sum!(WARP_SUMS, local, lane, warp_in_block);
            if tid == 0 {
                let clamped = if sum > eps * eps { sum } else { eps * eps };
                unsafe {
                    *row_scale.get_unchecked_mut(row as usize) = 1.0 / sqrt_f32(clamped);
                }
            }
        }
    }

    #[kernel]
    pub fn row_scale_apply_kernel(
        x: &[f32],
        row_scale: &[f32],
        mut out: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
    ) {
        let index = thread::blockIdx_x() * MATRIX_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < rows * cols {
            let row = index / cols;
            unsafe {
                *out.get_unchecked_mut(index as usize) =
                    x[index as usize] * row_scale[row as usize];
            }
        }
    }

    #[kernel]
    pub fn row_scale_refine_kernel(
        u: &[f32],
        mut row_scale: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
        target_row_sq: f32,
        eps: f32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = tid / WARP_SIZE;

        if row < rows {
            let mut local = 0.0;
            let mut col = tid;
            while col < cols {
                let value = u[(row * cols + col) as usize];
                local += value * value;
                col += MATRIX_THREADS_PER_BLOCK;
            }

            let row_sq = block_sum!(WARP_SUMS, local, lane, warp_in_block);
            let row_sq = if row_sq > eps * eps {
                row_sq
            } else {
                eps * eps
            };
            if tid == 0 {
                unsafe {
                    let scale = row_scale.as_mut_ptr().add(row as usize);
                    *scale *= sqrt_f32(target_row_sq / row_sq);
                }
            }
        }
    }
}
