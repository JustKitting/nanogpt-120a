use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel};

use crate::mma::{
    NVFP4_PROJECTION_CTA_A_PACKS, NVFP4_PROJECTION_CTA_A_SCALES, NVFP4_PROJECTION_CTA_B_PACKS,
    NVFP4_PROJECTION_CTA_B_SCALES, NVFP4_PROJECTION_CTA_THREADS, Nvfp4FourSixMmaWeightTensor,
    Nvfp4ProjectionParams, nvfp4_projection_cta_nobias_kernel_body, projection_cta_grid_dim,
};
use crate::nvfp4::Nvfp4RowwiseDeviceTensor;

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
        self.module.lm_head_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: projection_cta_grid_dim(args.token_count, args.vocab_size),
                block_dim: (NVFP4_PROJECTION_CTA_THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
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

#[allow(static_mut_refs)]
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn lm_head_kernel(
        input_bytes: &[u8],
        input_scales: &[u8],
        input_global_scales: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        mut logits: DisjointSlice<f32>,
        params: LmHeadParams,
    ) {
        static mut A_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS> = SharedArray::UNINIT;
        static mut B_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS> = SharedArray::UNINIT;
        static mut A_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES> = SharedArray::UNINIT;
        static mut B_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES> = SharedArray::UNINIT;

        nvfp4_projection_cta_nobias_kernel_body(
            input_bytes,
            input_scales,
            input_global_scales,
            weight_bytes,
            weight_scales,
            &mut logits,
            Nvfp4ProjectionParams {
                token_count: params.token_count,
                input_dim: params.input_dim,
                output_dim: params.vocab_size,
                weight_global_scale: weight_global_scale[0],
                bias_global_scale: 0.0,
                residual_add: 0,
                activation: 0,
            },
            unsafe { &mut A_PACKS },
            unsafe { &mut B_PACKS },
            unsafe { &mut A_SCALES },
            unsafe { &mut B_SCALES },
        );
    }
}
