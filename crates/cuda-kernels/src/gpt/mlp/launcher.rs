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

macro_rules! projection_launcher {
    ($method:ident, $args:ty, $kernel:ident, $residual_add:expr, outputs($($out:ident),+)) => {
        pub fn $method(&self, args: $args) -> Result<(), DriverError> {
            self.module.$kernel(
                args.stream,
                projection_config(args.token_count, args.input_dim, args.output_dim),
                args.input.bytes, args.input.scales, args.input.global_scales,
                args.weight.bytes, args.weight.scales, args.bias.bytes, args.bias.scales,
                args.weight.global_scale, args.bias.global_scale, $(args.$out,)*
                projection_params(args.token_count, args.input_dim, args.output_dim, $residual_add),
            )
        }
    };
}

macro_rules! relu2_backward_launcher {
    ($method:ident, $args:ty, $kernel:ident) => {
        pub fn $method(&self, args: $args) -> Result<(), DriverError> {
            self.module.$kernel(
                args.stream, linear_config(args.len, kernels::RELU2_THREADS_PER_BLOCK),
                args.pre_activation, args.d_out, args.d_pre_activation, args.len,
            )
        }
    };
}

impl MlpModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    projection_launcher!(up_relu2, MlpUpRelu2Args<'_, '_>, mlp_projection_relu2_kernel, 0, outputs(pre_activation, out));
    projection_launcher!(down_residual, MlpDownResidualArgs<'_, '_>, mlp_projection_kernel, 1, outputs(residual));
    relu2_backward_launcher!(relu2_backward, Relu2BackwardArgs<'_, '_>, relu2_backward_kernel);
    relu2_backward_launcher!(relu2_backward_f16, Relu2BackwardF16Args<'_, '_>, relu2_backward_f16_kernel);
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
