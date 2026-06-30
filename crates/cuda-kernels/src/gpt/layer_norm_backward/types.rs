use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError};

use super::kernel::{THREADS_PER_BLOCK, kernels};
use super::param::{PARAM_THREADS_PER_BLOCK, kernels as param_kernels};
use crate::launch::grid_x_config;
use crate::nvfp4::Nvfp4DeviceTensor;

pub struct LayerNormBackwardInputArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub residual: &'a DeviceBuffer<u16>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub mean: &'a DeviceBuffer<f32>,
    pub inv_std: &'a DeviceBuffer<f32>,
    pub weight: Nvfp4DeviceTensor<'a>,
    pub d_residual: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
}

pub struct LayerNormBackwardInputF32Args<'a, 'out> {
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
    pub residual: &'a DeviceBuffer<u16>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub mean: &'a DeviceBuffer<f32>,
    pub inv_std: &'a DeviceBuffer<f32>,
    pub d_weight: &'out mut DeviceBuffer<f32>,
    pub d_bias: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
}

pub struct LayerNormBackwardParamF32Args<'a, 'out> {
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
            grid_x_config(args.row_count, THREADS_PER_BLOCK),
            args.residual,
            args.d_normalized,
            args.mean,
            args.inv_std,
            args.weight.bytes,
            args.weight.scales,
            args.weight.global_scale,
            args.d_residual,
            args.row_count,
            args.embedding_dim,
        )
    }

    pub fn backward_input_f32(
        &self,
        args: LayerNormBackwardInputF32Args<'_, '_>,
    ) -> Result<(), DriverError> {
        self.module.layer_norm_backward_input_f32_kernel(
            args.stream,
            grid_x_config(args.row_count, THREADS_PER_BLOCK),
            args.residual,
            args.d_normalized,
            args.mean,
            args.inv_std,
            args.weight.bytes,
            args.weight.scales,
            args.weight.global_scale,
            args.d_residual,
            args.row_count,
            args.embedding_dim,
        )
    }

    pub fn backward_params(
        &self,
        args: LayerNormBackwardParamArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.param_module.layer_norm_backward_params_kernel(
            args.stream,
            grid_x_config(args.embedding_dim, PARAM_THREADS_PER_BLOCK),
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

    pub fn backward_params_f32(
        &self,
        args: LayerNormBackwardParamF32Args<'_, '_>,
    ) -> Result<(), DriverError> {
        self.param_module.layer_norm_backward_params_f32_kernel(
            args.stream,
            grid_x_config(args.embedding_dim, PARAM_THREADS_PER_BLOCK),
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
