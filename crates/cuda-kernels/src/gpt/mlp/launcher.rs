use std::sync::Arc;

use cuda_core::{CudaModule, DriverError, LaunchConfig};

use super::args::{MlpDownResidualArgs, MlpUpRelu2Args};
use super::kernels;
use crate::mma::{
    NVFP4_PROJECTION_ACTIVATION_NONE, NVFP4_PROJECTION_THREADS_PER_BLOCK, Nvfp4ProjectionParams,
    projection_grid_dim,
};

pub struct MlpModule {
    module: kernels::LoadedModule,
}

impl MlpModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn up_relu2(&self, args: MlpUpRelu2Args<'_, '_>) -> Result<(), DriverError> {
        self.module.mlp_projection_relu2_kernel(
            args.stream,
            config(args.token_count, args.output_dim),
            args.input.bytes,
            args.input.scales,
            args.input.global_scales,
            args.weight.bytes,
            args.weight.scales,
            args.bias.bytes,
            args.bias.scales,
            args.pre_activation,
            args.out,
            up_params(&args),
        )
    }

    pub fn down_residual(&self, args: MlpDownResidualArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.mlp_projection_kernel(
            args.stream,
            config(args.token_count, args.output_dim),
            args.input.bytes,
            args.input.scales,
            args.input.global_scales,
            args.weight.bytes,
            args.weight.scales,
            args.bias.bytes,
            args.bias.scales,
            args.residual,
            down_params(&args),
        )
    }
}

fn config(token_count: u32, output_dim: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: projection_grid_dim(token_count, output_dim),
        block_dim: (NVFP4_PROJECTION_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn up_params(args: &MlpUpRelu2Args<'_, '_>) -> Nvfp4ProjectionParams {
    Nvfp4ProjectionParams {
        token_count: args.token_count,
        input_dim: args.input_dim,
        output_dim: args.output_dim,
        weight_global_scale: args.weight.global_scale,
        bias_global_scale: args.bias.global_scale,
        residual_add: 0,
        activation: NVFP4_PROJECTION_ACTIVATION_NONE,
    }
}

fn down_params(args: &MlpDownResidualArgs<'_, '_>) -> Nvfp4ProjectionParams {
    Nvfp4ProjectionParams {
        token_count: args.token_count,
        input_dim: args.input_dim,
        output_dim: args.output_dim,
        weight_global_scale: args.weight.global_scale,
        bias_global_scale: args.bias.global_scale,
        residual_add: 1,
        activation: NVFP4_PROJECTION_ACTIVATION_NONE,
    }
}
