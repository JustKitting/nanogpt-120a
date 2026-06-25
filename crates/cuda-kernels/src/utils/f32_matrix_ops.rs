use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::float_ptx::fma_f32;

const F32_OPS_THREADS_PER_BLOCK: u32 = 256;

pub struct F32Linear2Args<'a, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<f32>,
    pub b: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub len: u32,
    pub a_scale: f32,
    pub b_scale: f32,
}

pub struct F32AddScaledIdentityArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub src: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub dim: u32,
    pub scale: f32,
}

pub struct F32MatrixOpsModule {
    module: module::LoadedModule,
}

impl F32MatrixOpsModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: module::from_module(module)?,
        })
    }

    pub fn linear2(&self, args: F32Linear2Args<'_, '_>) -> Result<(), DriverError> {
        assert!(args.a.len() >= args.len as usize);
        assert!(args.b.len() >= args.len as usize);
        assert!(args.out.len() >= args.len as usize);

        self.module.f32_linear2_kernel(
            args.stream,
            linear_config(args.len),
            args.a,
            args.b,
            args.out,
            args.len,
            args.a_scale,
            args.b_scale,
        )
    }

    pub fn add_scaled_identity(
        &self,
        args: F32AddScaledIdentityArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let len = args.dim * args.dim;
        assert!(args.src.len() >= len as usize);
        assert!(args.out.len() >= len as usize);

        self.module.f32_add_scaled_identity_kernel(
            args.stream,
            linear_config(len),
            args.src,
            args.out,
            args.dim,
            args.scale,
        )
    }
}

fn linear_config(element_count: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (element_count.div_ceil(F32_OPS_THREADS_PER_BLOCK), 1, 1),
        block_dim: (F32_OPS_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cuda_module]
mod module {
    use super::*;

    #[kernel]
    pub fn f32_linear2_kernel(
        a: &[f32],
        b: &[f32],
        mut out: DisjointSlice<f32>,
        len: u32,
        a_scale: f32,
        b_scale: f32,
    ) {
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let stride = thread::gridDim_x() * thread::blockDim_x();
        while index < len {
            let i = index as usize;
            unsafe {
                *out.get_unchecked_mut(i) = fma_f32(a_scale, a[i], b_scale * b[i]);
            }
            index += stride;
        }
    }

    #[kernel]
    pub fn f32_add_scaled_identity_kernel(
        src: &[f32],
        mut out: DisjointSlice<f32>,
        dim: u32,
        scale: f32,
    ) {
        let len = dim * dim;
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let stride = thread::gridDim_x() * thread::blockDim_x();
        while index < len {
            let row = index / dim;
            let col = index - row * dim;
            let add = if row == col { scale } else { 0.0 };
            unsafe {
                *out.get_unchecked_mut(index as usize) = src[index as usize] + add;
            }
            index += stride;
        }
    }
}
