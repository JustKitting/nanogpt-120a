use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};

use crate::nvfp4::Nvfp4DeviceTensor;

use super::{GPT_LAYER_NORM_THREADS_PER_BLOCK, THREADS_PER_BLOCK, WARPS_PER_BLOCK, kernels};

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

pub struct GptLayerNormSaveResidualF16Args<'a, 'out> {
    pub stream: &'a CudaStream,
    pub residual: &'a DeviceBuffer<f32>,
    pub weight: Nvfp4DeviceTensor<'a>,
    pub bias: Nvfp4DeviceTensor<'a>,
    pub normalized: &'out mut DeviceBuffer<f32>,
    pub normalized_amax: &'out mut DeviceBuffer<f32>,
    pub mean: &'out mut DeviceBuffer<f32>,
    pub inv_std: &'out mut DeviceBuffer<f32>,
    pub residual_f16: &'out mut DeviceBuffer<u16>,
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

    pub fn gpt_layer_norm_save_residual_f16(
        &self,
        args: GptLayerNormSaveResidualF16Args<'_, '_>,
    ) -> Result<(), DriverError> {
        self.module.gpt_layer_norm_save_residual_f16_kernel(
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
            args.residual_f16,
            args.row_count,
            args.embedding_dim,
            args.epsilon,
        )
    }
}
