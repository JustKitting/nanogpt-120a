use super::args::{NextLatGeluArgs, NextLatGeluBackwardArgs, NextLatResidualAddArgs};
use super::launcher::{NEXTLAT_THREADS_PER_BLOCK, NextLatModule};
use crate::launch::linear_config;
use cuda_core::DriverError;

macro_rules! activation_launcher {
    ($method:ident, $args:ty, $kernel:ident, $($buffer:ident),+) => {
        pub fn $method(&self, args: $args) -> Result<(), DriverError> {
            self.activation.$kernel(
                args.stream,
                linear_config(args.len, NEXTLAT_THREADS_PER_BLOCK),
                $(args.$buffer,)* args.len,
            )
        }
    };
}

impl NextLatModule {
    activation_launcher!(gelu, NextLatGeluArgs<'_, '_>, nextlat_gelu_kernel, input, out);
    activation_launcher!(gelu_backward, NextLatGeluBackwardArgs<'_, '_>, nextlat_gelu_backward_kernel, input, d_out, d_input);
    activation_launcher!(residual_add, NextLatResidualAddArgs<'_, '_>, nextlat_residual_add_kernel, delta, residual, out);
}
