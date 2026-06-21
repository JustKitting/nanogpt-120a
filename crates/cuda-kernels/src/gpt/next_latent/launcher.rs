use super::args::{
    NextLatConcatArgs, NextLatGeluArgs, NextLatProjectionArgs, NextLatResidualAddArgs,
    NextLatShape, NextLatSmoothL1Args, projection_params,
};
use super::{activation_kernels, kernels, projection_kernels};
use crate::mma::{NVFP4_PROJECTION_CTA_THREADS, projection_cta_grid_dim};
use cuda_core::{CudaModule, DriverError, LaunchConfig};
use std::sync::Arc;
const NEXTLAT_THREADS_PER_BLOCK: u32 = 256;

pub struct NextLatModule {
    core: kernels::module::LoadedModule,
    projection: projection_kernels::module::LoadedModule,
    activation: activation_kernels::module::LoadedModule,
}

impl NextLatModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            core: kernels::module::from_module(module.clone())?,
            projection: projection_kernels::module::from_module(module.clone())?,
            activation: activation_kernels::module::from_module(module)?,
        })
    }

    pub fn concat_input(&self, args: NextLatConcatArgs<'_, '_>) -> Result<(), DriverError> {
        self.core.nextlat_concat_input_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.row_count, 1, 1),
                block_dim: (NEXTLAT_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.next_token_embeddings,
            args.current_states,
            args.out,
            NextLatShape {
                row_count: args.row_count,
                embedding_dim: args.embedding_dim,
                seq_len: 0,
                batch_size: 0,
                lambda: 0.0,
            },
        )
    }

    pub fn smooth_l1(&self, args: NextLatSmoothL1Args<'_, '_>) -> Result<(), DriverError> {
        self.core.nextlat_smooth_l1_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.batch_size, args.seq_len, 1),
                block_dim: (NEXTLAT_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.predicted_next_states,
            args.target_states,
            args.losses,
            args.d_predicted_next_states,
            NextLatShape {
                row_count: args.batch_size * args.seq_len,
                embedding_dim: args.embedding_dim,
                seq_len: args.seq_len,
                batch_size: args.batch_size,
                lambda: args.lambda,
            },
        )
    }

    pub fn projection(&self, args: NextLatProjectionArgs<'_, '_>) -> Result<(), DriverError> {
        self.projection.nextlat_projection_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: projection_cta_grid_dim(args.token_count, args.output_dim),
                block_dim: (NVFP4_PROJECTION_CTA_THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            args.input.bytes,
            args.input.scales,
            args.input.global_scales,
            args.weight.bytes,
            args.weight.scales,
            args.bias.bytes,
            args.bias.scales,
            args.weight.global_scale,
            args.bias.global_scale,
            args.out,
            projection_params(&args),
        )
    }

    pub fn gelu(&self, args: NextLatGeluArgs<'_, '_>) -> Result<(), DriverError> {
        self.activation.nextlat_gelu_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.len.div_ceil(NEXTLAT_THREADS_PER_BLOCK), 1, 1),
                block_dim: (NEXTLAT_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.input,
            args.out,
            args.len,
        )
    }

    pub fn residual_add(&self, args: NextLatResidualAddArgs<'_, '_>) -> Result<(), DriverError> {
        self.activation.nextlat_residual_add_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.len.div_ceil(NEXTLAT_THREADS_PER_BLOCK), 1, 1),
                block_dim: (NEXTLAT_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.delta,
            args.residual,
            args.out,
            args.len,
        )
    }
}
