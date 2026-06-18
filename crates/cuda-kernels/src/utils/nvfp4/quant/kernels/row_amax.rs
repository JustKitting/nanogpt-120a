use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::amax::{amax4_f32, max4_f32};
use crate::float_ptx::{abs_f32, max_f32};
use crate::warp_reduce::warp_max_f32;

use super::super::config::WARPS_PER_BLOCK;

pub(crate) const TENSOR_AMAX_VALUES_PER_BLOCK: u32 = 1024;

#[allow(static_mut_refs)]
#[cuda_module]
pub(crate) mod module {
    use super::*;

    static mut ROW_AMAX: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;
    static mut TENSOR_AMAX: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

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

    #[kernel]
    pub fn tensor_chunk_amax_f32_kernel(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        element_count: u32,
    ) {
        let chunk = thread::blockIdx_x();
        let thread = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = thread / 32;
        let base = chunk * TENSOR_AMAX_VALUES_PER_BLOCK;

        let stride = thread::blockDim_x();
        let i0 = base + thread;
        let i1 = i0 + stride;
        let i2 = i1 + stride;
        let i3 = i2 + stride;

        let local_amax = if base + TENSOR_AMAX_VALUES_PER_BLOCK <= element_count {
            amax4_f32(
                x[i0 as usize],
                x[i1 as usize],
                x[i2 as usize],
                x[i3 as usize],
            )
        } else {
            max4_f32(
                checked_abs_f32(x, i0, element_count),
                checked_abs_f32(x, i1, element_count),
                checked_abs_f32(x, i2, element_count),
                checked_abs_f32(x, i3, element_count),
            )
        };

        let warp_amax = warp_max_f32(local_amax);
        if lane == 0 {
            unsafe {
                TENSOR_AMAX[warp_in_block as usize] = warp_amax;
            }
        }

        thread::sync_threads();

        if warp_in_block == 0 {
            let partial = if lane < WARPS_PER_BLOCK {
                unsafe { TENSOR_AMAX[lane as usize] }
            } else {
                0.0
            };
            let block_amax = warp_max_f32(partial);
            if lane == 0 {
                unsafe {
                    *out.get_unchecked_mut(chunk as usize) = block_amax;
                }
            }
        }
    }

    #[kernel]
    pub fn tensor_amax_from_chunks_f32_kernel(
        chunk_amax: &[f32],
        mut out: DisjointSlice<f32>,
        chunk_count: u32,
    ) {
        let thread = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = thread / 32;
        let mut chunk = thread;
        let mut local_amax = 0.0;

        while chunk < chunk_count {
            local_amax = max_f32(local_amax, chunk_amax[chunk as usize]);
            chunk += thread::blockDim_x();
        }

        let warp_amax = warp_max_f32(local_amax);
        if lane == 0 {
            unsafe {
                TENSOR_AMAX[warp_in_block as usize] = warp_amax;
            }
        }

        thread::sync_threads();

        if warp_in_block == 0 {
            let partial = if lane < WARPS_PER_BLOCK {
                unsafe { TENSOR_AMAX[lane as usize] }
            } else {
                0.0
            };
            let block_amax = warp_max_f32(partial);
            if lane == 0 {
                unsafe {
                    *out.get_unchecked_mut(0) = block_amax;
                }
            }
        }
    }

    #[inline(always)]
    fn checked_abs_f32(x: &[f32], index: u32, element_count: u32) -> f32 {
        if index < element_count {
            abs_f32(x[index as usize])
        } else {
            0.0
        }
    }
}
