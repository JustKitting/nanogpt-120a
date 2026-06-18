use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::float_ptx::sqrt_f32;

use super::super::super::threads::{WARP_SIZE, WARPS_PER_BLOCK};
use super::{reduce, scale};

const POLAR_EXPRESS_NORM_SAFETY: f32 = 1.01;
const POLAR_EXPRESS_EPS: f32 = 1.0e-7;

#[allow(static_mut_refs)]
#[cuda_module]
pub(crate) mod module {
    use super::*;

    #[kernel]
    pub fn polar_chunk_sum_kernel(x: &[f32], mut chunks: DisjointSlice<f32>, len: u32) {
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = tid / WARP_SIZE;
        let base = thread::blockIdx_x() * crate::optimizer::POLAR_SUM_VALUES_PER_BLOCK as u32;
        let local = reduce::input_chunk_sum(x, base, tid, len);
        let sum = block_sum!(WARP_SUMS, local, lane, warp_in_block);

        if tid == 0 {
            unsafe {
                *chunks.get_unchecked_mut(thread::blockIdx_x() as usize) = sum;
            }
        }
    }

    #[kernel]
    pub fn polar_inv_norm_from_chunks_kernel(
        chunks: &[f32],
        mut inv_norm: DisjointSlice<f32>,
        chunk_count: u32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = tid / WARP_SIZE;
        let local = reduce::chunk_sum(chunks, tid, chunk_count);
        let sum = block_sum!(WARP_SUMS, local, lane, warp_in_block);

        if tid == 0 {
            unsafe {
                *inv_norm.get_unchecked_mut(0) =
                    1.0 / (sqrt_f32(sum) * POLAR_EXPRESS_NORM_SAFETY + POLAR_EXPRESS_EPS);
            }
        }
    }

    #[kernel]
    pub fn polar_scale_from_inv_norm_kernel(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        inv_norm: &[f32],
        len: u32,
    ) {
        scale::store_scaled(x, &mut out, inv_norm[0], len);
    }

    #[kernel]
    pub fn polar_scale_in_place_from_inv_norm_kernel(
        mut x: DisjointSlice<f32>,
        inv_norm: &[f32],
        len: u32,
    ) {
        scale::scale_in_place(&mut x, inv_norm[0], len);
    }
}
