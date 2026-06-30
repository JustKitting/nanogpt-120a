use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

use super::DECODE_THREADS_PER_BLOCK;
use super::args::{Nvfp4DecodeTransposeArgs, Nvfp4RowwiseDecodeTransposeArgs};
use super::kernels;
use crate::launch::linear_config;

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
            linear_config(
                args.rows.saturating_mul(args.cols),
                DECODE_THREADS_PER_BLOCK,
            ),
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
            linear_config(
                args.rows.saturating_mul(args.cols),
                DECODE_THREADS_PER_BLOCK,
            ),
            args.input.bytes,
            args.input.scales,
            args.input.global_scales,
            args.output,
            args.rows,
            args.cols,
        )
    }
}
