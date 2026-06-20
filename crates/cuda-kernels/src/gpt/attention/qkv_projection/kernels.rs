use cuda_device::{DisjointSlice, cuda_module, kernel};

use crate::mma::{
    NVFP4_PROJECTION_CTA_A_PACKS, NVFP4_PROJECTION_CTA_A_SCALES, NVFP4_PROJECTION_CTA_B_PACKS,
    NVFP4_PROJECTION_CTA_B_SCALES, Nvfp4ProjectionParams, nvfp4_projection_cta_kernel_body,
};
use cuda_device::SharedArray;

#[allow(static_mut_refs)]
#[cuda_module]
mod module {
    use super::*;

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn attention_projection_kernel(
        input_bytes: &[u8],
        input_scales: &[u8],
        input_global_scales: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        bias_bytes: &[u8],
        bias_scales: &[u8],
        weight_global_scale: &[f32],
        bias_global_scale: &[f32],
        mut out: DisjointSlice<f32>,
        params: Nvfp4ProjectionParams,
    ) {
        let params = Nvfp4ProjectionParams {
            weight_global_scale: weight_global_scale[0],
            bias_global_scale: bias_global_scale[0],
            ..params
        };

        static mut A_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS> = SharedArray::UNINIT;
        static mut B_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS> = SharedArray::UNINIT;
        static mut A_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES> = SharedArray::UNINIT;
        static mut B_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES> = SharedArray::UNINIT;

        nvfp4_projection_cta_kernel_body(
            input_bytes,
            input_scales,
            input_global_scales,
            weight_bytes,
            weight_scales,
            bias_bytes,
            bias_scales,
            &mut out,
            params,
            unsafe { &mut A_PACKS },
            unsafe { &mut B_PACKS },
            unsafe { &mut A_SCALES },
            unsafe { &mut B_SCALES },
        );
    }
}

pub(crate) use module::{LoadedModule, from_module};
