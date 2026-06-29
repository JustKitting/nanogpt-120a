use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const TRANSPOSE_THREADS_PER_BLOCK: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TransposeParams {
    pub rows: u32,
    pub cols: u32,
}

unsafe impl DeviceCopy for TransposeParams {}

pub struct TransposeF32Args<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: &'a DeviceBuffer<f32>,
    pub output: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
    pub cols: u32,
}

pub struct TransposeModule {
    module: kernels::LoadedModule,
}

impl TransposeModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn transpose_f32(&self, args: TransposeF32Args<'_, '_>) -> Result<(), DriverError> {
        self.module.transpose_f32_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (
                    args.rows
                        .saturating_mul(args.cols)
                        .div_ceil(TRANSPOSE_THREADS_PER_BLOCK),
                    1,
                    1,
                ),
                block_dim: (TRANSPOSE_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.input,
            args.output,
            TransposeParams {
                rows: args.rows,
                cols: args.cols,
            },
        )
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn transpose_f32_kernel(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        params: TransposeParams,
    ) {
        let index = thread::blockIdx_x() * TRANSPOSE_THREADS_PER_BLOCK + thread::threadIdx_x();
        let len = params.rows * params.cols;
        if index < len {
            let out_index = (index % params.cols) * params.rows + index / params.cols;

            unsafe {
                *output.get_unchecked_mut(out_index as usize) = input[index as usize];
            }
        }
    }
}
