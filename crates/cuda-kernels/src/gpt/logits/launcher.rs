use std::sync::Arc;

use cuda_core::{CudaModule, DriverError, LaunchConfig};

use super::args::{ARGMAX_THREADS_PER_BLOCK, LogitsArgmaxArgs, LogitsArgmaxParams};
use super::kernels::kernels;

pub struct LogitsModule {
    module: kernels::LoadedModule,
}

impl LogitsModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn argmax(&self, args: LogitsArgmaxArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.logits_argmax_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (ARGMAX_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.logits,
            args.out_token,
            LogitsArgmaxParams {
                row: args.row,
                vocab_size: args.vocab_size,
            },
        )
    }
}
