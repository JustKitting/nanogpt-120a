use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::float_ptx::sqrt_f32;
use crate::layer_norm_utils::{
    centered_column, layer_norm_columns3, layer_norm_map3, layer_norm_map3_indexed,
    layer_norm_square_sum3, layer_norm_store3, layer_norm_sum3, max_abs3,
    nvfp4_affine_normalized_column, nvfp4_column,
};
use crate::nvfp4::Nvfp4DeviceTensor;
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

            let cols = layer_norm_columns3!(thread, EMBEDDING_THREADS_PER_BLOCK);
            let values = layer_norm_map3!(cols, |col| nvfp4_column(
                token_embedding_bytes,
                token_embedding_scales,
                params.token_embedding_global_scale,
                token_base,
                col,
                params.embedding_dim,
            ));

            let local_sum = layer_norm_sum3!(values);
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

            let centered = layer_norm_map3_indexed!(cols, |index, col| centered_column(
                col,
                params.embedding_dim,
                values[index],
                mean
            ));
            let local_variance_sum = layer_norm_square_sum3!(centered);
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

            let normalized_values =
                layer_norm_map3_indexed!(cols, |index, col| nvfp4_affine_normalized_column(
                    layer_norm_weight_bytes,
                    layer_norm_weight_scales,
                    layer_norm_bias_bytes,
                    layer_norm_bias_scales,
                    col,
                    params.embedding_dim,
                    centered[index],
                    inv_std,
                    params.layer_norm_weight_global_scale,
                    params.layer_norm_bias_global_scale,
                ));

            layer_norm_store3!(&mut residual, row_base, cols, params.embedding_dim, values);
            layer_norm_store3!(
                &mut normalized,
                row_base,
                cols,
                params.embedding_dim,
                normalized_values
            );

            let local_amax = max_abs3(
                normalized_values[0],
                normalized_values[1],
                normalized_values[2],
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
}
