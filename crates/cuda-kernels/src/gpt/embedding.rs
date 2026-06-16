use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::float_ptx::{abs_f32, fma_f32, max_f32, sqrt_f32};
use crate::nvfp4::{Nvfp4DeviceTensor, nvfp4_value};
use crate::warp_reduce::{warp_max_f32, warp_sum_f32};

const EMBEDDING_THREADS_PER_BLOCK: u32 = 256;
const WARP_SIZE: u32 = 32;
const WARPS_PER_BLOCK: u32 = EMBEDDING_THREADS_PER_BLOCK / WARP_SIZE;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmbeddingParams {
    pub hidden_len: u32,
    pub embedding_dim: u32,
    pub token_embedding_global_scale: f32,
    pub layer_norm_weight_global_scale: f32,
    pub layer_norm_bias_global_scale: f32,
    pub epsilon: f32,
}

unsafe impl DeviceCopy for EmbeddingParams {}

pub struct EmbeddingArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub tokens: &'a DeviceBuffer<u32>,
    pub token_embedding: Nvfp4DeviceTensor<'a>,
    pub layer_norm_weight: Nvfp4DeviceTensor<'a>,
    pub layer_norm_bias: Nvfp4DeviceTensor<'a>,
    pub residual: &'out mut DeviceBuffer<f32>,
    pub normalized: &'out mut DeviceBuffer<f32>,
    pub normalized_amax: &'out mut DeviceBuffer<f32>,
    pub hidden_len: u32,
    pub embedding_dim: u32,
    pub epsilon: f32,
}

pub struct EmbeddingModule {
    module: kernels::LoadedModule,
}

impl EmbeddingModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn token_embedding_layer_norm(
        &self,
        args: EmbeddingArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.module.token_embedding_layer_norm_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.hidden_len / args.embedding_dim, 1, 1),
                block_dim: (EMBEDDING_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.tokens,
            args.token_embedding.bytes,
            args.token_embedding.scales,
            args.layer_norm_weight.bytes,
            args.layer_norm_weight.scales,
            args.layer_norm_bias.bytes,
            args.layer_norm_bias.scales,
            args.residual,
            args.normalized,
            args.normalized_amax,
            EmbeddingParams {
                hidden_len: args.hidden_len,
                embedding_dim: args.embedding_dim,
                token_embedding_global_scale: args.token_embedding.global_scale,
                layer_norm_weight_global_scale: args.layer_norm_weight.global_scale,
                layer_norm_bias_global_scale: args.layer_norm_bias.global_scale,
                epsilon: args.epsilon,
            },
        )
    }
}

#[allow(static_mut_refs)]
#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn token_embedding_layer_norm_kernel(
        tokens: &[u32],
        token_embedding_bytes: &[u8],
        token_embedding_scales: &[u8],
        layer_norm_weight_bytes: &[u8],
        layer_norm_weight_scales: &[u8],
        layer_norm_bias_bytes: &[u8],
        layer_norm_bias_scales: &[u8],
        mut residual: DisjointSlice<f32>,
        mut normalized: DisjointSlice<f32>,
        mut normalized_amax: DisjointSlice<f32>,
        params: EmbeddingParams,
    ) {
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let thread = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = thread / WARP_SIZE;

        if row < params.hidden_len / params.embedding_dim {
            let token = tokens[row as usize];
            let row_base = row as usize * params.embedding_dim as usize;
            let token_base = token as usize * params.embedding_dim as usize;

            let col0 = thread;
            let col1 = thread + EMBEDDING_THREADS_PER_BLOCK;
            let col2 = thread + EMBEDDING_THREADS_PER_BLOCK * 2;

            macro_rules! load_token_column {
                ($col:expr) => {
                    token_value(
                        token_embedding_bytes,
                        token_embedding_scales,
                        params.token_embedding_global_scale,
                        token_base,
                        $col,
                        params.embedding_dim,
                    )
                };
            }

            let value0 = load_token_column!(col0);
            let value1 = load_token_column!(col1);
            let value2 = load_token_column!(col2);

            let local_sum = value0 + value1 + value2;
            let warp_total = warp_sum_f32(local_sum);

            if lane == 0 {
                unsafe {
                    WARP_SUMS[warp_in_block as usize] = warp_total;
                }
            }

            thread::sync_threads();

            if warp_in_block == 0 {
                let partial = if lane < WARPS_PER_BLOCK {
                    unsafe { WARP_SUMS[lane as usize] }
                } else {
                    0.0
                };
                let block_total = warp_sum_f32(partial);

                if lane == 0 {
                    unsafe {
                        WARP_SUMS[0] = block_total / params.embedding_dim as f32;
                    }
                }
            }

            thread::sync_threads();

            let mean = unsafe { WARP_SUMS[0] };

            macro_rules! centered_column {
                ($col:expr, $value:expr) => {
                    if $col < params.embedding_dim {
                        $value - mean
                    } else {
                        0.0
                    }
                };
            }

            let centered0 = centered_column!(col0, value0);
            let centered1 = centered_column!(col1, value1);
            let centered2 = centered_column!(col2, value2);

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
                let partial = if lane < WARPS_PER_BLOCK {
                    unsafe { WARP_SUMS[lane as usize] }
                } else {
                    0.0
                };
                let block_total = warp_sum_f32(partial);

                if lane == 0 {
                    unsafe {
                        WARP_SUMS[0] = 1.0
                            / sqrt_f32(block_total / params.embedding_dim as f32 + params.epsilon);
                    }
                }
            }

            thread::sync_threads();

            let inv_std = unsafe { WARP_SUMS[0] };

            macro_rules! normalized_column {
                ($col:expr, $centered:expr) => {
                    if $col < params.embedding_dim {
                        let weight = layer_norm_value(
                            layer_norm_weight_bytes,
                            layer_norm_weight_scales,
                            params.layer_norm_weight_global_scale,
                            $col,
                        );
                        let bias = layer_norm_value(
                            layer_norm_bias_bytes,
                            layer_norm_bias_scales,
                            params.layer_norm_bias_global_scale,
                            $col,
                        );

                        fma_f32($centered * inv_std, weight, bias)
                    } else {
                        0.0
                    }
                };
            }

            let normalized0 = normalized_column!(col0, centered0);
            let normalized1 = normalized_column!(col1, centered1);
            let normalized2 = normalized_column!(col2, centered2);

            macro_rules! store_hidden_column {
                ($col:expr, $value:expr, $normalized_value:expr) => {
                    if $col < params.embedding_dim {
                        unsafe {
                            *residual.get_unchecked_mut(row_base + $col as usize) = $value;
                            *normalized.get_unchecked_mut(row_base + $col as usize) =
                                $normalized_value;
                        }
                    }
                };
            }

            store_hidden_column!(col0, value0, normalized0);
            store_hidden_column!(col1, value1, normalized1);
            store_hidden_column!(col2, value2, normalized2);

            let local_amax = max_f32(
                abs_f32(normalized0),
                max_f32(abs_f32(normalized1), abs_f32(normalized2)),
            );
            let warp_amax = warp_max_f32(local_amax);

            if lane == 0 {
                unsafe {
                    WARP_SUMS[warp_in_block as usize] = warp_amax;
                }
            }

            thread::sync_threads();

            if warp_in_block == 0 {
                let partial = if lane < WARPS_PER_BLOCK {
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

    #[inline(always)]
    fn token_value(
        bytes: &[u8],
        scales: &[u8],
        global_scale: f32,
        token_base: usize,
        col: u32,
        embedding_dim: u32,
    ) -> f32 {
        if col < embedding_dim {
            nvfp4_value(bytes, scales, global_scale, token_base + col as usize)
        } else {
            0.0
        }
    }

    #[inline(always)]
    fn layer_norm_value(
        layer_norm_bytes: &[u8],
        layer_norm_scales: &[u8],
        layer_norm_global_scale: f32,
        col: u32,
    ) -> f32 {
        nvfp4_value(
            layer_norm_bytes,
            layer_norm_scales,
            layer_norm_global_scale,
            col as usize,
        )
    }
}
