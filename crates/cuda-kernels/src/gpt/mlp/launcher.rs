use std::sync::Arc;

use cuda_core::{CudaModule, DriverError, LaunchConfig};

use super::args::{MlpDownResidualArgs, MlpUpRelu2Args, Relu2BackwardArgs, Relu2BackwardF16Args};
use super::kernels;
use crate::mma::{
    NVFP4_PROJECTION_ACTIVATION_NONE, NVFP4_PROJECTION_CTA_THREADS, Nvfp4ProjectionParams,
    projection_cta_grid_dim, projection_cta_row_pair_grid_dim,
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
            aligned_config(args.token_count, args.input_dim, args.output_dim),
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
            up_params(&args),
        )
    }

    pub fn down_residual(&self, args: MlpDownResidualArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.mlp_projection_kernel(
            args.stream,
            aligned_config(args.token_count, args.input_dim, args.output_dim),
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
            down_params(&args),
        )
    }

    pub fn relu2_backward(&self, args: Relu2BackwardArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.relu2_backward_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.len.div_ceil(kernels::RELU2_THREADS_PER_BLOCK), 1, 1),
                block_dim: (kernels::RELU2_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
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
            LaunchConfig {
                grid_dim: (args.len.div_ceil(kernels::RELU2_THREADS_PER_BLOCK), 1, 1),
                block_dim: (kernels::RELU2_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.pre_activation,
            args.d_out,
            args.d_pre_activation,
            args.len,
        )
    }
}

fn aligned_config(token_count: u32, input_dim: u32, output_dim: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: if projection_cta_aligned(token_count, input_dim, output_dim) {
            projection_cta_row_pair_grid_dim(token_count, output_dim)
        } else {
            projection_cta_grid_dim(token_count, output_dim)
        },
        block_dim: (NVFP4_PROJECTION_CTA_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn projection_cta_aligned(token_count: u32, input_dim: u32, output_dim: u32) -> bool {
    use crate::mma::{NVFP4_PROJECTION_CTA_K, NVFP4_PROJECTION_CTA_M, NVFP4_PROJECTION_CTA_N};

    token_count % NVFP4_PROJECTION_CTA_M == 0
        && input_dim % NVFP4_PROJECTION_CTA_K == 0
        && output_dim % NVFP4_PROJECTION_CTA_N == 0
}

fn up_params(args: &MlpUpRelu2Args<'_, '_>) -> Nvfp4ProjectionParams {
    Nvfp4ProjectionParams {
        token_count: args.token_count,
        input_dim: args.input_dim,
        output_dim: args.output_dim,
        weight_global_scale: 1.0,
        bias_global_scale: 1.0,
        residual_add: 0,
        activation: NVFP4_PROJECTION_ACTIVATION_NONE,
    }
}

fn down_params(args: &MlpDownResidualArgs<'_, '_>) -> Nvfp4ProjectionParams {
    Nvfp4ProjectionParams {
        token_count: args.token_count,
        input_dim: args.input_dim,
        output_dim: args.output_dim,
        weight_global_scale: 1.0,
        bias_global_scale: 1.0,
        residual_add: 1,
        activation: NVFP4_PROJECTION_ACTIVATION_NONE,
    }
}
