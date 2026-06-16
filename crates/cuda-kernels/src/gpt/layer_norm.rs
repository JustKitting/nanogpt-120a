use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::float_ptx::{fma_f32, sqrt_f32};
use crate::layer_norm_utils::{
    centered_column, f32_column, max_abs3, nvfp4_affine_normalized_column, store_column,
};
use crate::nvfp4::Nvfp4DeviceTensor;
use crate::warp_reduce::{warp_max_f32, warp_sum_f32};

pub const ROW_SIZE: usize = 32;
const WARPS_PER_BLOCK: u32 = 8;
const THREADS_PER_BLOCK: u32 = WARPS_PER_BLOCK * ROW_SIZE as u32;
const GPT_LAYER_NORM_THREADS_PER_BLOCK: u32 = 256;
const WARP_SIZE: u32 = 32;
const GPT_LAYER_NORM_WARPS_PER_BLOCK: u32 = GPT_LAYER_NORM_THREADS_PER_BLOCK / WARP_SIZE;

#[allow(static_mut_refs)]
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

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn gpt_layer_norm_kernel(
        residual: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        bias_bytes: &[u8],
        bias_scales: &[u8],
        mut normalized: DisjointSlice<f32>,
        mut normalized_amax: DisjointSlice<f32>,
        row_count: u32,
        embedding_dim: u32,
        weight_global_scale: f32,
        bias_global_scale: f32,
        epsilon: f32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, { GPT_LAYER_NORM_WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let thread = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = thread / WARP_SIZE;

        if row < row_count {
            let row_base = row as usize * embedding_dim as usize;
            let col0 = thread;
            let col1 = thread + GPT_LAYER_NORM_THREADS_PER_BLOCK;
            let col2 = thread + GPT_LAYER_NORM_THREADS_PER_BLOCK * 2;

            let value0 = f32_column(residual, row_base, col0, embedding_dim);
            let value1 = f32_column(residual, row_base, col1, embedding_dim);
            let value2 = f32_column(residual, row_base, col2, embedding_dim);

            let local_sum = value0 + value1 + value2;
            let warp_total = warp_sum_f32(local_sum);

            if lane == 0 {
                unsafe {
                    WARP_SUMS[warp_in_block as usize] = warp_total;
                }
            }

            thread::sync_threads();

            if warp_in_block == 0 {
                let partial = if lane < GPT_LAYER_NORM_WARPS_PER_BLOCK {
                    unsafe { WARP_SUMS[lane as usize] }
                } else {
                    0.0
                };
                let block_total = warp_sum_f32(partial);

                if lane == 0 {
                    unsafe {
                        WARP_SUMS[0] = block_total / embedding_dim as f32;
                    }
                }
            }

            thread::sync_threads();

            let mean = unsafe { WARP_SUMS[0] };
            let centered0 = centered_column(col0, embedding_dim, value0, mean);
            let centered1 = centered_column(col1, embedding_dim, value1, mean);
            let centered2 = centered_column(col2, embedding_dim, value2, mean);

            let local_variance_sum =
                centered0 * centered0 + centered1 * centered1 + centered2 * centered2;
            let warp_total = warp_sum_f32(local_variance_sum);

            if lane == 0 {
                unsafe {
                    WARP_SUMS[warp_in_block as usize] = warp_total;
                }
            }

            thread::sync_threads();

            if warp_in_block == 0 {
                let partial = if lane < GPT_LAYER_NORM_WARPS_PER_BLOCK {
                    unsafe { WARP_SUMS[lane as usize] }
                } else {
                    0.0
                };
                let block_total = warp_sum_f32(partial);

                if lane == 0 {
                    unsafe {
                        WARP_SUMS[0] = 1.0 / sqrt_f32(block_total / embedding_dim as f32 + epsilon);
                    }
                }
            }

            thread::sync_threads();

            let inv_std = unsafe { WARP_SUMS[0] };
            let normalized0 = nvfp4_affine_normalized_column(
                weight_bytes,
                weight_scales,
                bias_bytes,
                bias_scales,
                col0,
                embedding_dim,
                centered0,
                inv_std,
                weight_global_scale,
                bias_global_scale,
            );
            let normalized1 = nvfp4_affine_normalized_column(
                weight_bytes,
                weight_scales,
                bias_bytes,
                bias_scales,
                col1,
                embedding_dim,
                centered1,
                inv_std,
                weight_global_scale,
                bias_global_scale,
            );
            let normalized2 = nvfp4_affine_normalized_column(
                weight_bytes,
                weight_scales,
                bias_bytes,
                bias_scales,
                col2,
                embedding_dim,
                centered2,
                inv_std,
                weight_global_scale,
                bias_global_scale,
            );

            store_column(&mut normalized, row_base, col0, embedding_dim, normalized0);
            store_column(&mut normalized, row_base, col1, embedding_dim, normalized1);
            store_column(&mut normalized, row_base, col2, embedding_dim, normalized2);

            let local_amax = max_abs3(normalized0, normalized1, normalized2);
            let warp_amax = warp_max_f32(local_amax);

            if lane == 0 {
                unsafe {
                    WARP_SUMS[warp_in_block as usize] = warp_amax;
                }
            }

            thread::sync_threads();

            if warp_in_block == 0 {
                let partial = if lane < GPT_LAYER_NORM_WARPS_PER_BLOCK {
                    unsafe { WARP_SUMS[lane as usize] }
                } else {
                    0.0
                };
                let block_amax = warp_max_f32(partial);

                if lane == 0 {
                    unsafe {
                        *normalized_amax.get_unchecked_mut(row as usize) = block_amax;
                    }
                }
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

pub struct GptLayerNormArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub residual: &'a DeviceBuffer<f32>,
    pub weight: Nvfp4DeviceTensor<'a>,
    pub bias: Nvfp4DeviceTensor<'a>,
    pub normalized: &'out mut DeviceBuffer<f32>,
    pub normalized_amax: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
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

    pub fn gpt_layer_norm(&self, args: GptLayerNormArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.gpt_layer_norm_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.row_count, 1, 1),
                block_dim: (GPT_LAYER_NORM_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.residual,
            args.weight.bytes,
            args.weight.scales,
            args.bias.bytes,
            args.bias.scales,
            args.normalized,
            args.normalized_amax,
            args.row_count,
            args.embedding_dim,
            args.weight.global_scale,
            args.bias.global_scale,
            args.epsilon,
        )
    }
}
