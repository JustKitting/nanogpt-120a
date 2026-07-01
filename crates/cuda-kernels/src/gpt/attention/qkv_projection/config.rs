use cuda_core::LaunchConfig;

use super::args::QkvProjectionArgs;
use crate::launch::launch_config;
use crate::mma::{NVFP4_PROJECTION_CTA_THREADS, Nvfp4ProjectionParams, projection_cta_grid_dim};

pub(super) fn cta_config(token_count: u32, output_dim: u32) -> LaunchConfig {
    launch_config(
        projection_cta_grid_dim(token_count, output_dim),
        NVFP4_PROJECTION_CTA_THREADS,
    )
}

pub(super) fn qkv_params(args: &QkvProjectionArgs<'_, '_>) -> Nvfp4ProjectionParams {
    Nvfp4ProjectionParams::new(args.token_count, args.input_dim, args.output_dim)
}

pub(super) fn c_proj_params(token_count: u32, embedding_dim: u32) -> Nvfp4ProjectionParams {
    Nvfp4ProjectionParams::new(token_count, embedding_dim, embedding_dim).with_residual_add(1)
}
