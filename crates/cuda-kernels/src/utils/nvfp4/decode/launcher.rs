use std::sync::Arc;

use cuda_core::{CudaModule, DriverError, LaunchConfig};

use super::DECODE_THREADS_PER_BLOCK;
use super::args::{Nvfp4DecodeTransposeArgs, Nvfp4RowwiseDecodeTransposeArgs};
use super::kernels;

pub struct Nvfp4DecodeModule {
    module: kernels::LoadedModule,
}

impl Nvfp4DecodeModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn decode_transpose_f32(
        &self,
        args: Nvfp4DecodeTransposeArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.module.nvfp4_decode_transpose_f32_kernel(
            args.stream,
            config(args.rows, args.cols),
            args.input.bytes,
            args.input.scales,
            args.input.global_scale,
            args.output,
            args.rows,
            args.cols,
        )
    }

    pub fn decode_rowwise_transpose_f32(
        &self,
        args: Nvfp4RowwiseDecodeTransposeArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.module.nvfp4_decode_rowwise_transpose_f32_kernel(
            args.stream,
            config(args.rows, args.cols),
            args.input.bytes,
            args.input.scales,
            args.input.global_scales,
            args.output,
            args.rows,
            args.cols,
        )
    }
}

fn config(rows: u32, cols: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (
            rows.saturating_mul(cols).div_ceil(DECODE_THREADS_PER_BLOCK),
            1,
            1,
        ),
        block_dim: (DECODE_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}
