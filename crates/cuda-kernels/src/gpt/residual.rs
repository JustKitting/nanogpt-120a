use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::launch::linear_config;

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

macro_rules! residual_launcher {
    ($method:ident, $args:ty, $kernel:ident, $($buffer:ident),+) => {
        pub fn $method(&self, args: $args) -> Result<(), DriverError> {
            self.module.$kernel(
                args.stream, linear_config(args.len, THREADS_PER_BLOCK),
                $(args.$buffer,)* args.out, args.len,
            )
        }
    };
}

impl ResidualBackwardModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    residual_launcher!(
        grad_add,
        ResidualGradAddArgs<'_, '_>,
        residual_grad_add_kernel,
        direct,
        branch
    );
    residual_launcher!(
        grad_accumulate,
        ResidualGradAccumulateArgs<'_, '_>,
        residual_grad_accumulate_kernel,
        branch
    );
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
        if let Some(index) = residual_index(len) {
            unsafe {
                *out.get_unchecked_mut(index) = direct[index] + branch[index];
            }
        }
    }

    #[kernel]
    pub fn residual_grad_accumulate_kernel(branch: &[f32], mut out: DisjointSlice<f32>, len: u32) {
        if let Some(index) = residual_index(len) {
            unsafe {
                let slot = out.get_unchecked_mut(index);
                *slot += branch[index];
            }
        }
    }

    #[inline(always)]
    fn residual_index(len: u32) -> Option<usize> {
        let index = thread::blockIdx_x() * THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            Some(index as usize)
        } else {
            None
        }
    }
}
