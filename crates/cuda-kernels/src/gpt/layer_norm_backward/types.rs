use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};

use super::kernel::{THREADS_PER_BLOCK, kernels};
use super::param::{PARAM_THREADS_PER_BLOCK, kernels as param_kernels};
use crate::nvfp4::Nvfp4DeviceTensor;

pub struct LayerNormBackwardInputArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub residual: &'a DeviceBuffer<f32>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub mean: &'a DeviceBuffer<f32>,
    pub inv_std: &'a DeviceBuffer<f32>,
    pub weight: Nvfp4DeviceTensor<'a>,
    pub d_residual: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
}

pub struct LayerNormBackwardParamArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub residual: &'a DeviceBuffer<f32>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub mean: &'a DeviceBuffer<f32>,
    pub inv_std: &'a DeviceBuffer<f32>,
    pub d_weight: &'out mut DeviceBuffer<f32>,
    pub d_bias: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
}

pub struct LayerNormBackwardModule {
    module: kernels::LoadedModule,
    param_module: param_kernels::LoadedModule,
}

impl LayerNormBackwardModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module.clone())?,
            param_module: param_kernels::from_module(module)?,
        })
    }

    pub fn backward_input(
        &self,
        args: LayerNormBackwardInputArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.module.layer_norm_backward_input_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.row_count, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.residual,
            args.d_normalized,
            args.mean,
            args.inv_std,
            args.weight.bytes,
            args.weight.scales,
            args.d_residual,
            args.row_count,
            args.embedding_dim,
            args.weight.global_scale,
        )
    }

    pub fn backward_params(
        &self,
        args: LayerNormBackwardParamArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.param_module.layer_norm_backward_params_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.embedding_dim, 1, 1),
                block_dim: (PARAM_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.residual,
            args.d_normalized,
            args.mean,
            args.inv_std,
            args.d_weight,
            args.d_bias,
            args.row_count,
            args.embedding_dim,
        )
    }
}
