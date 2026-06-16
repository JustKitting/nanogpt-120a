use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::float_ptx::{abs_f32, max_f32};
use crate::warp_reduce::warp_max_f32;

use super::super::config::WARPS_PER_BLOCK;

#[allow(static_mut_refs)]
#[cuda_module]
pub(crate) mod module {
    use super::*;

    static mut ROW_AMAX: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

    #[kernel]
    pub fn row_amax_f32_kernel(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        row_count: u32,
        row_len: u32,
    ) {
        let row = thread::blockIdx_x();
        let thread = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = thread / 32;

        if row < row_count {
            let row_base = row as usize * row_len as usize;
            let mut col = thread;
            let mut local_amax = 0.0;

            while col < row_len {
                local_amax = max_f32(local_amax, abs_f32(x[row_base + col as usize]));
                col += thread::blockDim_x();
            }

            let warp_amax = warp_max_f32(local_amax);
            if lane == 0 {
                unsafe {
                    ROW_AMAX[warp_in_block as usize] = warp_amax;
                }
            }

            thread::sync_threads();

            if warp_in_block == 0 {
                let partial = if lane < WARPS_PER_BLOCK {
                    unsafe { ROW_AMAX[lane as usize] }
                } else {
                    0.0
                };
                let block_amax = warp_max_f32(partial);
                if lane == 0 {
                    unsafe {
                        *out.get_unchecked_mut(row as usize) = block_amax;
                    }
                }
            }
        }
    }
}
