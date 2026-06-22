use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const THREADS_PER_BLOCK: u32 = 256;

pub struct ResidualGradAddArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub direct: &'a DeviceBuffer<f32>,
    pub branch: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub len: u32,
}

pub struct ResidualGradAccumulateArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub branch: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub len: u32,
}

pub struct ResidualBackwardModule {
    module: kernels::LoadedModule,
}

impl ResidualBackwardModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn grad_add(&self, args: ResidualGradAddArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.residual_grad_add_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.len.div_ceil(THREADS_PER_BLOCK), 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.direct,
            args.branch,
            args.out,
            args.len,
        )
    }

    pub fn grad_accumulate(
        &self,
        args: ResidualGradAccumulateArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.module.residual_grad_accumulate_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.len.div_ceil(THREADS_PER_BLOCK), 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.branch,
            args.out,
            args.len,
        )
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn residual_grad_add_kernel(
        direct: &[f32],
        branch: &[f32],
        mut out: DisjointSlice<f32>,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            unsafe {
                *out.get_unchecked_mut(index as usize) =
                    direct[index as usize] + branch[index as usize];
            }
        }
    }

    #[kernel]
    pub fn residual_grad_accumulate_kernel(branch: &[f32], mut out: DisjointSlice<f32>, len: u32) {
        let index = thread::blockIdx_x() * THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            unsafe {
                let slot = out.get_unchecked_mut(index as usize);
                *slot += branch[index as usize];
            }
        }
    }
}
