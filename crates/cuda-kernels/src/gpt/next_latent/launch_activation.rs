use super::args::{NextLatGeluArgs, NextLatGeluBackwardArgs, NextLatResidualAddArgs};
use super::launcher::{NEXTLAT_THREADS_PER_BLOCK, NextLatModule};
use crate::launch::linear_config;
use cuda_core::DriverError;

impl NextLatModule {
    pub fn gelu(&self, args: NextLatGeluArgs<'_, '_>) -> Result<(), DriverError> {
        self.activation.nextlat_gelu_kernel(
            args.stream,
            linear_config(args.len, NEXTLAT_THREADS_PER_BLOCK),
            args.input,
            args.out,
            args.len,
        )
    }

    pub fn gelu_backward(&self, args: NextLatGeluBackwardArgs<'_, '_>) -> Result<(), DriverError> {
        self.activation.nextlat_gelu_backward_kernel(
            args.stream,
            linear_config(args.len, NEXTLAT_THREADS_PER_BLOCK),
            args.input,
            args.d_out,
            args.d_input,
            args.len,
        )
    }

    pub fn residual_add(&self, args: NextLatResidualAddArgs<'_, '_>) -> Result<(), DriverError> {
        self.activation.nextlat_residual_add_kernel(
            args.stream,
            linear_config(args.len, NEXTLAT_THREADS_PER_BLOCK),
            args.delta,
            args.residual,
            args.out,
            args.len,
        )
    }
}
