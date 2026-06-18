use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::float_ptx::sqrt_f32;

use super::{MATRIX_THREADS_PER_BLOCK, WARP_SIZE, WARPS_PER_BLOCK};

#[allow(static_mut_refs)]
#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn matrix_frobenius_norm_kernel(x: &[f32], mut out: DisjointSlice<f32>, len: u32) {
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = tid / WARP_SIZE;
        let mut local = 0.0;
        let mut index = tid;

        while index < len {
            let value = x[index as usize];
            local += value * value;
            index += MATRIX_THREADS_PER_BLOCK;
        }

        let sum = block_sum!(WARP_SUMS, local, lane, warp_in_block);
        if tid == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = sqrt_f32(sum) + 1.0e-7;
            }
        }
    }

    #[kernel]
    pub fn matrix_scale_in_place_kernel(mut x: DisjointSlice<f32>, norm: &[f32], len: u32) {
        let index = thread::blockIdx_x() * MATRIX_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            unsafe {
                let value = x.get_unchecked_mut(index as usize);
                *value /= norm[0];
            }
        }
    }

    #[kernel]
    pub fn matrix_combine_kernel(
        a: &[f32],
        b: &[f32],
        mut out: DisjointSlice<f32>,
        a_scale: f32,
        b_scale: f32,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * MATRIX_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            unsafe {
                *out.get_unchecked_mut(index as usize) =
                    a_scale * a[index as usize] + b_scale * b[index as usize];
            }
        }
    }
}
