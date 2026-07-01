mod args;

use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

use super::kernel::{THREADS_PER_BLOCK, kernels};
use super::param::{PARAM_THREADS_PER_BLOCK, kernels as param_kernels};
use crate::launch::grid_x_config;

pub use args::{
    LayerNormBackwardInputArgs, LayerNormBackwardInputF32Args, LayerNormBackwardParamArgs,
    LayerNormBackwardParamF32Args,
};

pub struct LayerNormBackwardModule {
    module: kernels::LoadedModule,
    param_module: param_kernels::LoadedModule,
}

macro_rules! backward_input_launcher {
    ($method:ident, $args:ty, $kernel:ident) => {
        pub fn $method(&self, args: $args) -> Result<(), DriverError> {
            self.module.$kernel(
                args.stream,
                grid_x_config(args.row_count, THREADS_PER_BLOCK),
                args.residual, args.d_normalized, args.mean, args.inv_std,
                args.weight.bytes, args.weight.scales, args.weight.global_scale,
                args.d_residual, args.row_count, args.embedding_dim,
            )
        }
    };
}

macro_rules! backward_params_launcher {
    ($method:ident, $args:ty, $kernel:ident) => {
        pub fn $method(&self, args: $args) -> Result<(), DriverError> {
            self.param_module.$kernel(
                args.stream,
                grid_x_config(args.embedding_dim, PARAM_THREADS_PER_BLOCK),
                args.residual, args.d_normalized, args.mean, args.inv_std,
                args.d_weight, args.d_bias, args.row_count, args.embedding_dim,
            )
        }
    };
}

impl LayerNormBackwardModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module.clone())?,
            param_module: param_kernels::from_module(module)?,
        })
    }

    backward_input_launcher!(backward_input, LayerNormBackwardInputArgs<'_, '_>, layer_norm_backward_input_kernel);
    backward_input_launcher!(backward_input_f32, LayerNormBackwardInputF32Args<'_, '_>, layer_norm_backward_input_f32_kernel);
    backward_params_launcher!(backward_params, LayerNormBackwardParamArgs<'_, '_>, layer_norm_backward_params_kernel);
    backward_params_launcher!(backward_params_f32, LayerNormBackwardParamF32Args<'_, '_>, layer_norm_backward_params_f32_kernel);
}
