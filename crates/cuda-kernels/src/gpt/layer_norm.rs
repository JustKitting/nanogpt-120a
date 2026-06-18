use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::float_ptx::{fma_f32, sqrt_f32};
use crate::layer_norm_reduce::{layer_norm_block_reduce, layer_norm_store_row};
use crate::layer_norm_utils::{
    centered_column, f32_column, layer_norm_columns3, layer_norm_map3, layer_norm_map3_indexed,
    layer_norm_square_sum3, layer_norm_store3, layer_norm_sum3, max_abs3,
    nvfp4_affine_normalized_column,
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
        weight_global_scale: &[f32],
        bias_global_scale: &[f32],
        mut normalized: DisjointSlice<f32>,
        mut normalized_amax: DisjointSlice<f32>,
        mut mean_out: DisjointSlice<f32>,
        mut inv_std_out: DisjointSlice<f32>,
        row_count: u32,
        embedding_dim: u32,
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
            let cols = layer_norm_columns3!(thread, GPT_LAYER_NORM_THREADS_PER_BLOCK);
            let values = layer_norm_map3!(cols, |col| f32_column(
                residual,
                row_base,
                col,
                embedding_dim
            ));

            let mean = layer_norm_block_reduce!(
                WARP_SUMS,
                GPT_LAYER_NORM_WARPS_PER_BLOCK,
                layer_norm_sum3!(values),
                lane,
                warp_in_block,
                warp_sum_f32
            ) / embedding_dim as f32;
            layer_norm_store_row!(&mut mean_out, row, lane, warp_in_block, mean);
            let centered = layer_norm_map3_indexed!(cols, |index, col| centered_column(
                col,
                embedding_dim,
                values[index],
                mean
            ));
            let variance_sum = layer_norm_block_reduce!(
                WARP_SUMS,
                GPT_LAYER_NORM_WARPS_PER_BLOCK,
                layer_norm_square_sum3!(centered),
                lane,
                warp_in_block,
                warp_sum_f32
            );
            let inv_std = 1.0 / sqrt_f32(variance_sum / embedding_dim as f32 + epsilon);
            layer_norm_store_row!(&mut inv_std_out, row, lane, warp_in_block, inv_std);
            let normalized_values =
                layer_norm_map3_indexed!(cols, |index, col| nvfp4_affine_normalized_column(
                    weight_bytes,
                    weight_scales,
                    bias_bytes,
                    bias_scales,
                    col,
                    embedding_dim,
                    centered[index],
                    inv_std,
                    weight_global_scale[0],
                    bias_global_scale[0],
                ));

            layer_norm_store3!(
                &mut normalized,
                row_base,
                cols,
                embedding_dim,
                normalized_values
            );

            let local_amax = max_abs3(
                normalized_values[0],
                normalized_values[1],
                normalized_values[2],
            );
            let block_amax = layer_norm_block_reduce!(
                WARP_SUMS,
                GPT_LAYER_NORM_WARPS_PER_BLOCK,
                local_amax,
                lane,
                warp_in_block,
                warp_max_f32
            );

            layer_norm_store_row!(&mut normalized_amax, row, lane, warp_in_block, block_amax);
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
    pub mean: &'out mut DeviceBuffer<f32>,
    pub inv_std: &'out mut DeviceBuffer<f32>,
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
            args.weight.global_scale,
            args.bias.global_scale,
            args.normalized,
            args.normalized_amax,
            args.mean,
            args.inv_std,
            args.row_count,
            args.embedding_dim,
            args.epsilon,
        )
    }
}
