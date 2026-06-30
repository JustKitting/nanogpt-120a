use super::args::{NextLatProjectionArgs, projection_params};
use super::launcher::NextLatModule;
use crate::launch::launch_config;
use crate::mma::{NVFP4_PROJECTION_CTA_THREADS, projection_cta_launch_grid_dim};
use cuda_core::DriverError;

impl NextLatModule {
    pub fn projection(&self, args: NextLatProjectionArgs<'_, '_>) -> Result<(), DriverError> {
        self.projection.nextlat_projection_kernel(
            args.stream,
            launch_config(
                projection_cta_launch_grid_dim(args.token_count, args.input_dim, args.output_dim),
                NVFP4_PROJECTION_CTA_THREADS,
            ),
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
