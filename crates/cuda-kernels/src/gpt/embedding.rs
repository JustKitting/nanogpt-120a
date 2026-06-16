use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::kernel_ops::{sqrt_f32, warp_sum_f32};
use crate::nvfp4::nvfp4_value;

const EMBEDDING_THREADS_PER_BLOCK: u32 = 256;
const WARP_SIZE: u32 = 32;
const WARPS_PER_BLOCK: u32 = EMBEDDING_THREADS_PER_BLOCK / WARP_SIZE;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmbeddingParams {
    pub hidden_len: u32,
    pub embedding_dim: u32,
    pub token_embedding_global_scale: f32,
    pub rms_weight_global_scale: f32,
    pub epsilon: f32,
}

unsafe impl DeviceCopy for EmbeddingParams {}

pub struct Nvfp4DeviceTensor<'a> {
    pub bytes: &'a DeviceBuffer<u8>,
    pub scales: &'a DeviceBuffer<u8>,
    pub global_scale: f32,
}

pub struct EmbeddingArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub tokens: &'a DeviceBuffer<u32>,
    pub token_embedding: Nvfp4DeviceTensor<'a>,
    pub rms_weight: Nvfp4DeviceTensor<'a>,
    pub residual: &'out mut DeviceBuffer<f32>,
    pub normalized: &'out mut DeviceBuffer<f32>,
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

    pub fn token_embedding_rmsnorm(&self, args: EmbeddingArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.token_embedding_rmsnorm_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.hidden_len / args.embedding_dim, 1, 1),
                block_dim: (EMBEDDING_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.tokens,
            args.token_embedding.bytes,
            args.token_embedding.scales,
            args.rms_weight.bytes,
            args.rms_weight.scales,
            args.residual,
            args.normalized,
            EmbeddingParams {
                hidden_len: args.hidden_len,
                embedding_dim: args.embedding_dim,
                token_embedding_global_scale: args.token_embedding.global_scale,
                rms_weight_global_scale: args.rms_weight.global_scale,
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
    pub fn token_embedding_rmsnorm_kernel(
        tokens: &[u32],
        token_embedding_bytes: &[u8],
        token_embedding_scales: &[u8],
        rms_weight_bytes: &[u8],
        rms_weight_scales: &[u8],
        mut residual: DisjointSlice<f32>,
        mut normalized: DisjointSlice<f32>,
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

            let local_sum = value0 * value0 + value1 * value1 + value2 * value2;
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
                        WARP_SUMS[0] = 1.0
                            / sqrt_f32(block_total / params.embedding_dim as f32 + params.epsilon);
                    }
                }
            }

            thread::sync_threads();

            let inv_rms = unsafe { WARP_SUMS[0] };

            macro_rules! store_hidden_column {
                ($col:expr, $value:expr) => {
                    if $col < params.embedding_dim {
                        let weight = rms_weight_value(
                            rms_weight_bytes,
                            rms_weight_scales,
                            params.rms_weight_global_scale,
                            $col,
                        );

                        unsafe {
                            *residual.get_unchecked_mut(row_base + $col as usize) = $value;
                            *normalized.get_unchecked_mut(row_base + $col as usize) =
                                $value * inv_rms * weight;
                        }
                    }
                };
            }

            store_hidden_column!(col0, value0);
            store_hidden_column!(col1, value1);
            store_hidden_column!(col2, value2);
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
    fn rms_weight_value(
        rms_weight_bytes: &[u8],
        rms_weight_scales: &[u8],
        rms_weight_global_scale: f32,
        col: u32,
    ) -> f32 {
        nvfp4_value(
            rms_weight_bytes,
            rms_weight_scales,
            rms_weight_global_scale,
            col as usize,
        )
    }
}
