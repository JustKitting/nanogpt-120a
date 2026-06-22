use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::float_ptx::{abs_f32, exp_f32, ln_f32, max_f32};
use crate::warp_reduce::{warp_max_f32, warp_sum_f32};

const CROSS_ENTROPY_THREADS_PER_BLOCK: u32 = 1024;
const WARP_SIZE: u32 = 32;
const CROSS_ENTROPY_WARPS_PER_BLOCK: u32 = CROSS_ENTROPY_THREADS_PER_BLOCK / WARP_SIZE;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CrossEntropyParams {
    pub token_count: u32,
    pub vocab_size: u32,
}

unsafe impl DeviceCopy for CrossEntropyParams {}

pub struct CrossEntropyArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub logits: &'a DeviceBuffer<f32>,
    pub targets: &'a DeviceBuffer<u32>,
    pub losses: &'out mut DeviceBuffer<f32>,
    pub dlogits: &'out mut DeviceBuffer<f32>,
    pub dlogits_row_amax: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub vocab_size: u32,
}

pub struct LossModule {
    module: kernels::LoadedModule,
}

impl LossModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn cross_entropy(&self, args: CrossEntropyArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.cross_entropy_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.token_count, 1, 1),
                block_dim: (CROSS_ENTROPY_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.logits,
            args.targets,
            args.losses,
            args.dlogits,
            args.dlogits_row_amax,
            CrossEntropyParams {
                token_count: args.token_count,
                vocab_size: args.vocab_size,
            },
        )
    }
}

#[allow(static_mut_refs)]
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn cross_entropy_kernel(
        logits: &[f32],
        targets: &[u32],
        mut losses: DisjointSlice<f32>,
        mut dlogits: DisjointSlice<f32>,
        mut dlogits_row_amax: DisjointSlice<f32>,
        params: CrossEntropyParams,
    ) {
        static mut REDUCE: SharedArray<f32, { CROSS_ENTROPY_WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let thread = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = thread / WARP_SIZE;

        if row < params.token_count {
            let row_base = row as usize * params.vocab_size as usize;
            let mut local_max = f32::NEG_INFINITY;
            let mut col = thread;
            while col < params.vocab_size {
                local_max = max_f32(local_max, logits[row_base + col as usize]);
                col += CROSS_ENTROPY_THREADS_PER_BLOCK;
            }

            let warp_max = warp_max_f32(local_max);
            if lane == 0 {
                unsafe {
                    REDUCE[warp_in_block as usize] = warp_max;
                }
            }

            thread::sync_threads();

            if warp_in_block == 0 {
                let partial = if lane < CROSS_ENTROPY_WARPS_PER_BLOCK {
                    unsafe { REDUCE[lane as usize] }
                } else {
                    f32::NEG_INFINITY
                };
                let block_max = warp_max_f32(partial);
                if lane == 0 {
                    unsafe {
                        REDUCE[0] = block_max;
                    }
                }
            }

            thread::sync_threads();

            let row_max = unsafe { REDUCE[0] };
            let mut local_sum = 0.0_f32;
            col = thread;
            while col < params.vocab_size {
                local_sum += exp_f32(logits[row_base + col as usize] - row_max);
                col += CROSS_ENTROPY_THREADS_PER_BLOCK;
            }

            let warp_sum = warp_sum_f32(local_sum);
            if lane == 0 {
                unsafe {
                    REDUCE[warp_in_block as usize] = warp_sum;
                }
            }

            thread::sync_threads();

            if warp_in_block == 0 {
                let partial = if lane < CROSS_ENTROPY_WARPS_PER_BLOCK {
                    unsafe { REDUCE[lane as usize] }
                } else {
                    0.0
                };
                let block_sum = warp_sum_f32(partial);
                if lane == 0 {
                    unsafe {
                        REDUCE[0] = block_sum;
                    }
                }
            }

            thread::sync_threads();

            let denom = unsafe { REDUCE[0] };
            let target = targets[row as usize];
            if thread == 0 {
                let target_logit = logits[row_base + target as usize];
                unsafe {
                    *losses.get_unchecked_mut(row as usize) =
                        ln_f32(denom) + row_max - target_logit;
                }
            }

            let mut local_dlogits_amax = 0.0_f32;
            col = thread;
            while col < params.vocab_size {
                let probability = exp_f32(logits[row_base + col as usize] - row_max) / denom;
                let target_delta = if col == target { 1.0 } else { 0.0 };
                let grad_scale = 1.0 / params.token_count as f32;
                let dlogit = (probability - target_delta) * grad_scale;
                local_dlogits_amax = max_f32(local_dlogits_amax, abs_f32(dlogit));
                unsafe {
                    *dlogits.get_unchecked_mut(row_base + col as usize) = dlogit;
                }
                col += CROSS_ENTROPY_THREADS_PER_BLOCK;
            }

            let warp_dlogits_amax = warp_max_f32(local_dlogits_amax);
            if lane == 0 {
                unsafe {
                    REDUCE[warp_in_block as usize] = warp_dlogits_amax;
                }
            }

            thread::sync_threads();

            if warp_in_block == 0 {
                let partial = if lane < CROSS_ENTROPY_WARPS_PER_BLOCK {
                    unsafe { REDUCE[lane as usize] }
                } else {
                    0.0
                };
                let row_dlogits_amax = warp_max_f32(partial);
                if lane == 0 {
                    unsafe {
                        *dlogits_row_amax.get_unchecked_mut(row as usize) = row_dlogits_amax;
                    }
                }
            }
        }
    }
}
