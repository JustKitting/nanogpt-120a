use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DeviceCopy, DriverError};

use crate::launch::launch_config;
use crate::mma::{
    NVFP4_PROJECTION_CTA_THREADS, Nvfp4FourSixMmaWeightTensor, projection_cta_grid_dim,
    projection_cta_row_pair_grid_dim,
};
use crate::nvfp4::Nvfp4RowwiseDeviceTensor;

#[path = "lm_head/kernels.rs"]
mod kernels;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LmHeadParams {
    pub token_count: u32,
    pub input_dim: u32,
    pub vocab_size: u32,
    pub weight_global_scale: f32,
}

unsafe impl DeviceCopy for LmHeadParams {}

pub struct LmHeadArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub logits: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub input_dim: u32,
    pub vocab_size: u32,
}

pub struct LmHeadModule {
    module: kernels::LoadedModule,
}

impl LmHeadModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn logits(&self, args: LmHeadArgs<'_, '_>) -> Result<(), DriverError> {
        let grid_dim = if lm_head_cta_aligned(args.token_count, args.input_dim, args.vocab_size) {
            projection_cta_row_pair_grid_dim(args.token_count, args.vocab_size)
        } else {
            projection_cta_grid_dim(args.token_count, args.vocab_size)
        };

        self.module.lm_head_kernel(
            args.stream,
            launch_config(grid_dim, NVFP4_PROJECTION_CTA_THREADS),
            args.input.bytes,
            args.input.scales,
            args.input.global_scales,
            args.weight.bytes,
            args.weight.scales,
            args.weight.global_scale,
            args.logits,
            LmHeadParams {
                token_count: args.token_count,
                input_dim: args.input_dim,
                vocab_size: args.vocab_size,
                weight_global_scale: 1.0,
            },
        )
    }
}

fn lm_head_cta_aligned(token_count: u32, input_dim: u32, vocab_size: u32) -> bool {
    use crate::mma::{NVFP4_PROJECTION_CTA_K, NVFP4_PROJECTION_CTA_M, NVFP4_PROJECTION_CTA_N};

    token_count % NVFP4_PROJECTION_CTA_M == 0
        && vocab_size % NVFP4_PROJECTION_CTA_N == 0
        && input_dim % NVFP4_PROJECTION_CTA_K == 0
}
