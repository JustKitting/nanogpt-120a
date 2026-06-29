use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::mma::{
    NVFP4_PROJECTION_CTA_A_PACKS, NVFP4_PROJECTION_CTA_A_SCALES, NVFP4_PROJECTION_CTA_B_PACKS,
    NVFP4_PROJECTION_CTA_B_SCALES, NVFP4_PROJECTION_CTA_K, NVFP4_PROJECTION_CTA_M,
    NVFP4_PROJECTION_CTA_N, Nvfp4ProjectionCtaTile, Nvfp4ProjectionParams,
    nvfp4_projection_cta_kernel_body, nvfp4_projection_cta_kernel_body_at_aligned_row_pair,
};

#[allow(static_mut_refs)]
#[cuda_module]
pub mod module {
    use super::*;

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn nextlat_projection_kernel(
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
        static mut A_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS> = SharedArray::UNINIT;
        static mut A1_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS> = SharedArray::UNINIT;
        static mut B_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS> = SharedArray::UNINIT;
        static mut A_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES> = SharedArray::UNINIT;
        static mut A1_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES> = SharedArray::UNINIT;
        static mut B_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES> = SharedArray::UNINIT;

        let params = Nvfp4ProjectionParams {
            weight_global_scale: weight_global_scale[0],
            bias_global_scale: bias_global_scale[0],
            ..params
        };

        if projection_cta_aligned(params) {
            let tile_col = thread::blockIdx_x();
            let tile_row_pair = thread::blockIdx_y();
            let thread_id = thread::threadIdx_x();
            let tile0 =
                Nvfp4ProjectionCtaTile::from_grid_tile(tile_col, tile_row_pair * 2, thread_id);
            let tile1 =
                Nvfp4ProjectionCtaTile::from_grid_tile(tile_col, tile_row_pair * 2 + 1, thread_id);
            nvfp4_projection_cta_kernel_body_at_aligned_row_pair(
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
                unsafe { &mut A1_PACKS },
                unsafe { &mut B_PACKS },
                unsafe { &mut A_SCALES },
                unsafe { &mut A1_SCALES },
                unsafe { &mut B_SCALES },
                tile0,
                tile1,
            );
        } else {
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

    #[inline(always)]
    fn projection_cta_aligned(params: Nvfp4ProjectionParams) -> bool {
        params.token_count % NVFP4_PROJECTION_CTA_M == 0
            && params.input_dim % NVFP4_PROJECTION_CTA_K == 0
            && params.output_dim % NVFP4_PROJECTION_CTA_N == 0
    }
}
