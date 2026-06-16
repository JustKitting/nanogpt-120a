use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread, warp};

use crate::kernel_ops::{fma_f32, sqrt_f32, warp_sum_f32};

pub const ROW_SIZE: usize = 32;
const WARPS_PER_BLOCK: u32 = 8;
const THREADS_PER_BLOCK: u32 = WARPS_PER_BLOCK * ROW_SIZE as u32;

#[cuda_module]
mod kernels {
    use super::*;

    const ROW_SIZE_F32: f32 = ROW_SIZE as f32;

    #[kernel]
    pub fn layer_norm_warp_f32_kernel(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mut out: DisjointSlice<f32>,
        row_count: u32,
        epsilon: f32,
    ) {
        let lane = warp::lane_id() as usize;
        let warp_in_block = thread::threadIdx_x() / ROW_SIZE as u32;
        let warps_per_block = thread::blockDim_x() / ROW_SIZE as u32;
        let row = thread::blockIdx_x() * warps_per_block + warp_in_block;

        if row < row_count {
            let index = row as usize * ROW_SIZE + lane;
            let value = x[index];
            let mean = warp_sum_f32(value) / ROW_SIZE_F32;
            let centered = value - mean;
            let variance = warp_sum_f32(centered * centered) / ROW_SIZE_F32;
            let inv_std = 1.0 / sqrt_f32(variance + epsilon);
            let normalized = centered * inv_std;

            unsafe {
                *out.get_unchecked_mut(index) = fma_f32(normalized, gamma[lane], beta[lane]);
            }
        }
    }
}

pub struct LayerNormArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub x: &'a DeviceBuffer<f32>,
    pub gamma: &'a DeviceBuffer<f32>,
    pub beta: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub epsilon: f32,
}

pub struct LayerNormModule {
    module: kernels::LoadedModule,
}

impl LayerNormModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn layer_norm_warp_f32(&self, args: LayerNormArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.layer_norm_warp_f32_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.row_count.div_ceil(WARPS_PER_BLOCK), 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.x,
            args.gamma,
            args.beta,
            args.out,
            args.row_count,
            args.epsilon,
        )
    }
}
