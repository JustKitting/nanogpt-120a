use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::float_ptx::sqrt_f32;

use super::super::super::threads::{WARP_SIZE, WARPS_PER_BLOCK};
use super::normalize;

const POLAR_EXPRESS_NORM_SAFETY: f32 = 1.01;
const POLAR_EXPRESS_EPS: f32 = 1.0e-7;

#[allow(static_mut_refs)]
#[cuda_module]
pub(crate) mod module {
    use super::*;

    #[kernel]
    pub fn polar_normalize_kernel(x: &[f32], mut out: DisjointSlice<f32>, len: u32) {
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = tid / WARP_SIZE;
        let local = normalize::frobenius_local_sum(x, tid, len);
        let sum = block_sum!(WARP_SUMS, local, lane, warp_in_block);
        let inv_norm = 1.0 / (sqrt_f32(sum) * POLAR_EXPRESS_NORM_SAFETY + POLAR_EXPRESS_EPS);

        thread::sync_threads();
        normalize::store_normalized(x, &mut out, inv_norm, tid, len);
    }

    #[kernel]
    pub fn polar_normalize_in_place_kernel(mut x: DisjointSlice<f32>, len: u32) {
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = tid / WARP_SIZE;
        let local = normalize::frobenius_local_sum_disjoint(&mut x, tid, len);
        let sum = block_sum!(WARP_SUMS, local, lane, warp_in_block);
        let inv_norm = 1.0 / (sqrt_f32(sum) * POLAR_EXPRESS_NORM_SAFETY + POLAR_EXPRESS_EPS);

        thread::sync_threads();
        normalize::scale_normalized(&mut x, inv_norm, tid, len);
    }
}
