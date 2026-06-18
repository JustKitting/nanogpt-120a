use cuda_core::LaunchConfig;

use super::args::QkvProjectionArgs;
use crate::mma::{
    NVFP4_PROJECTION_ACTIVATION_NONE, NVFP4_PROJECTION_THREADS_PER_BLOCK, Nvfp4ProjectionParams,
    projection_grid_dim,
};

pub(super) fn config(token_count: u32, output_dim: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: projection_grid_dim(token_count, output_dim),
        block_dim: (NVFP4_PROJECTION_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub(super) fn qkv_params(args: &QkvProjectionArgs<'_, '_>) -> Nvfp4ProjectionParams {
    projection_params(
        args.token_count,
        args.input_dim,
        args.output_dim,
        1.0,
        1.0,
        0,
    )
}

pub(super) fn c_proj_params(token_count: u32, embedding_dim: u32) -> Nvfp4ProjectionParams {
    projection_params(token_count, embedding_dim, embedding_dim, 1.0, 1.0, 1)
}

fn projection_params(
    token_count: u32,
    input_dim: u32,
    output_dim: u32,
    weight_global_scale: f32,
    bias_global_scale: f32,
    residual_add: u32,
) -> Nvfp4ProjectionParams {
    Nvfp4ProjectionParams {
        token_count,
        input_dim,
        output_dim,
        weight_global_scale,
        bias_global_scale,
        residual_add,
        activation: NVFP4_PROJECTION_ACTIVATION_NONE,
    }
}
