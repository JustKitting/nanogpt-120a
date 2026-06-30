use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

use super::args::{MlpDownResidualArgs, MlpUpRelu2Args, Relu2BackwardArgs, Relu2BackwardF16Args};
use super::kernels;
use crate::launch::{launch_config, linear_config};
use crate::mma::{
    NVFP4_PROJECTION_ACTIVATION_NONE, NVFP4_PROJECTION_CTA_THREADS, Nvfp4ProjectionParams,
    projection_cta_launch_grid_dim,
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
            projection_config(args.token_count, args.input_dim, args.output_dim),
            args.input.bytes,
            args.input.scales,
            args.input.global_scales,
            args.weight.bytes,
            args.weight.scales,
            args.bias.bytes,
            args.bias.scales,
            args.weight.global_scale,
            args.bias.global_scale,
            args.pre_activation,
            args.out,
            projection_params(args.token_count, args.input_dim, args.output_dim, 0),
        )
    }

    pub fn down_residual(&self, args: MlpDownResidualArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.mlp_projection_kernel(
            args.stream,
            projection_config(args.token_count, args.input_dim, args.output_dim),
            args.input.bytes,
            args.input.scales,
            args.input.global_scales,
            args.weight.bytes,
            args.weight.scales,
            args.bias.bytes,
            args.bias.scales,
            args.weight.global_scale,
            args.bias.global_scale,
            args.residual,
            projection_params(args.token_count, args.input_dim, args.output_dim, 1),
        )
    }

    pub fn relu2_backward(&self, args: Relu2BackwardArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.relu2_backward_kernel(
            args.stream,
            linear_config(args.len, kernels::RELU2_THREADS_PER_BLOCK),
            args.pre_activation,
            args.d_out,
            args.d_pre_activation,
            args.len,
        )
    }

    pub fn relu2_backward_f16(
        &self,
        args: Relu2BackwardF16Args<'_, '_>,
    ) -> Result<(), DriverError> {
        self.module.relu2_backward_f16_kernel(
            args.stream,
            linear_config(args.len, kernels::RELU2_THREADS_PER_BLOCK),
            args.pre_activation,
            args.d_out,
            args.d_pre_activation,
            args.len,
        )
    }
}

fn projection_config(token_count: u32, input_dim: u32, output_dim: u32) -> cuda_core::LaunchConfig {
    launch_config(
        projection_cta_launch_grid_dim(token_count, input_dim, output_dim),
        NVFP4_PROJECTION_CTA_THREADS,
    )
}

fn projection_params(
    token_count: u32,
    input_dim: u32,
    output_dim: u32,
    residual_add: u32,
) -> Nvfp4ProjectionParams {
    Nvfp4ProjectionParams {
        token_count,
        input_dim,
        output_dim,
        weight_global_scale: 1.0,
        bias_global_scale: 1.0,
        residual_add,
        activation: NVFP4_PROJECTION_ACTIVATION_NONE,
    }
}
