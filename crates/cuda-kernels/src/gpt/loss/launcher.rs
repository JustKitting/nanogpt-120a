use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};

use super::{CROSS_ENTROPY_THREADS_PER_BLOCK, CrossEntropyParams, kernels};

pub struct CrossEntropyArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub logits: &'a DeviceBuffer<f32>,
    pub targets: &'a DeviceBuffer<u32>,
    pub losses: &'out mut DeviceBuffer<f32>,
    pub dlogits: &'out mut DeviceBuffer<f32>,
    pub dlogits_row_amax: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub vocab_size: u32,
}

pub struct LossModule {
    module: kernels::LoadedModule,
}

impl LossModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn cross_entropy(&self, args: CrossEntropyArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.cross_entropy_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.token_count, 1, 1),
                block_dim: (CROSS_ENTROPY_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.logits,
            args.targets,
            args.losses,
            args.dlogits,
            args.dlogits_row_amax,
            CrossEntropyParams {
                token_count: args.token_count,
                vocab_size: args.vocab_size,
            },
        )
    }
}
