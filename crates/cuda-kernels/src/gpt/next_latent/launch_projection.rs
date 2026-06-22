use super::args::{NextLatProjectionArgs, projection_params};
use super::launcher::NextLatModule;
use crate::mma::{
    NVFP4_PROJECTION_CTA_THREADS, projection_cta_grid_dim, projection_cta_row_pair_grid_dim,
};
use cuda_core::{DriverError, LaunchConfig};

impl NextLatModule {
    pub fn projection(&self, args: NextLatProjectionArgs<'_, '_>) -> Result<(), DriverError> {
        self.projection.nextlat_projection_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: if projection_cta_aligned(
                    args.token_count,
                    args.input_dim,
                    args.output_dim,
                ) {
                    projection_cta_row_pair_grid_dim(args.token_count, args.output_dim)
                } else {
                    projection_cta_grid_dim(args.token_count, args.output_dim)
                },
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
}

fn projection_cta_aligned(token_count: u32, input_dim: u32, output_dim: u32) -> bool {
    use crate::mma::{NVFP4_PROJECTION_CTA_K, NVFP4_PROJECTION_CTA_M, NVFP4_PROJECTION_CTA_N};

    token_count % NVFP4_PROJECTION_CTA_M == 0
        && input_dim % NVFP4_PROJECTION_CTA_K == 0
        && output_dim % NVFP4_PROJECTION_CTA_N == 0
}
