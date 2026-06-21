use super::args::{NextLatGeluArgs, NextLatGeluBackwardArgs, NextLatResidualAddArgs};
use super::launcher::{NEXTLAT_THREADS_PER_BLOCK, NextLatModule};
use cuda_core::{DriverError, LaunchConfig};

impl NextLatModule {
    pub fn gelu(&self, args: NextLatGeluArgs<'_, '_>) -> Result<(), DriverError> {
        self.activation.nextlat_gelu_kernel(
            args.stream,
            config(args.len),
            args.input,
            args.out,
            args.len,
        )
    }

    pub fn gelu_backward(&self, args: NextLatGeluBackwardArgs<'_, '_>) -> Result<(), DriverError> {
        self.activation.nextlat_gelu_backward_kernel(
            args.stream,
            config(args.len),
            args.input,
            args.d_out,
            args.d_input,
            args.len,
        )
    }

    pub fn residual_add(&self, args: NextLatResidualAddArgs<'_, '_>) -> Result<(), DriverError> {
        self.activation.nextlat_residual_add_kernel(
            args.stream,
            config(args.len),
            args.delta,
            args.residual,
            args.out,
            args.len,
        )
    }
}

fn config(len: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (len.div_ceil(NEXTLAT_THREADS_PER_BLOCK), 1, 1),
        block_dim: (NEXTLAT_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}
