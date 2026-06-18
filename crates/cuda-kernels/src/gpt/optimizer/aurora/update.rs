use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::amax::max4_f32;
use crate::block_reduce::block_max_f32;

use super::super::threads::{APPLY_THREADS_PER_BLOCK, WARP_SIZE, WARPS_PER_BLOCK};
use update_math::update_average_abs;

#[path = "update_math.rs"]
mod update_math;

const UPDATE_AMAX_VALUES_PER_BLOCK: u32 = APPLY_THREADS_PER_BLOCK * 4;

#[allow(static_mut_refs)]
#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn fp32_weight_update_average_chunk_amax_kernel(
        mut z_master: DisjointSlice<f32>,
        mut x_master: DisjointSlice<f32>,
        aurora_update: &[f32],
        mut chunk_amax: DisjointSlice<f32>,
        learning_rate: f32,
        weight_decay: f32,
        average_coefficient: f32,
        len: u32,
    ) {
        static mut WARP_AMAX: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

        let chunk = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = tid / WARP_SIZE;
        let base = chunk * UPDATE_AMAX_VALUES_PER_BLOCK;
        let stride = APPLY_THREADS_PER_BLOCK;

        let local_amax = max4_f32(
            update_average_abs(
                &mut z_master,
                &mut x_master,
                aurora_update,
                learning_rate,
                weight_decay,
                average_coefficient,
                base + tid,
                len,
            ),
            update_average_abs(
                &mut z_master,
                &mut x_master,
                aurora_update,
                learning_rate,
                weight_decay,
                average_coefficient,
                base + tid + stride,
                len,
            ),
            update_average_abs(
                &mut z_master,
                &mut x_master,
                aurora_update,
                learning_rate,
                weight_decay,
                average_coefficient,
                base + tid + stride * 2,
                len,
            ),
            update_average_abs(
                &mut z_master,
                &mut x_master,
                aurora_update,
                learning_rate,
                weight_decay,
                average_coefficient,
                base + tid + stride * 3,
                len,
            ),
        );

        let block_amax =
            block_max_f32!(WARP_AMAX, local_amax, lane, warp_in_block, WARPS_PER_BLOCK);
        if tid == 0 {
            unsafe {
                *chunk_amax.get_unchecked_mut(chunk as usize) = block_amax;
            }
        }
    }
}
