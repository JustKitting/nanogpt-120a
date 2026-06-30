use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use super::LmHeadParams;
use crate::mma::{
    NVFP4_PROJECTION_CTA_A_PACKS, NVFP4_PROJECTION_CTA_A_SCALES, NVFP4_PROJECTION_CTA_B_PACKS,
    NVFP4_PROJECTION_CTA_B_SCALES, Nvfp4ProjectionCtaTile, Nvfp4ProjectionParams,
    nvfp4_projection_cta_nobias_kernel_body,
    nvfp4_projection_cta_nobias_kernel_body_at_aligned_row_pair, projection_cta_shape_aligned,
};

#[allow(static_mut_refs)]
#[cuda_module]
pub(super) mod module {
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
        static mut A1_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS> = SharedArray::UNINIT;
        static mut B_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS> = SharedArray::UNINIT;
        static mut A_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES> = SharedArray::UNINIT;
        static mut A1_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES> = SharedArray::UNINIT;
        static mut B_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES> = SharedArray::UNINIT;

        let projection = Nvfp4ProjectionParams {
            token_count: params.token_count,
            input_dim: params.input_dim,
            output_dim: params.vocab_size,
            weight_global_scale: weight_global_scale[0],
            bias_global_scale: 0.0,
            residual_add: 0,
            activation: 0,
        };

        if projection_cta_shape_aligned(params.token_count, params.input_dim, params.vocab_size) {
            let (tile0, tile1) = Nvfp4ProjectionCtaTile::row_pair(thread::threadIdx_x());

            nvfp4_projection_cta_nobias_kernel_body_at_aligned_row_pair(
                input_bytes,
                input_scales,
                input_global_scales,
                weight_bytes,
                weight_scales,
                &mut logits,
                projection,
                unsafe { &mut A_PACKS },
                unsafe { &mut A1_PACKS },
                unsafe { &mut B_PACKS },
                unsafe { &mut A_SCALES },
                unsafe { &mut A1_SCALES },
                unsafe { &mut B_SCALES },
                tile0,
                tile1,
            );
        } else {
            nvfp4_projection_cta_nobias_kernel_body(
                input_bytes,
                input_scales,
                input_global_scales,
                weight_bytes,
                weight_scales,
                &mut logits,
                projection,
                unsafe { &mut A_PACKS },
                unsafe { &mut B_PACKS },
                unsafe { &mut A_SCALES },
                unsafe { &mut B_SCALES },
            );
        }
    }
}

pub(crate) use module::{LoadedModule, from_module};
