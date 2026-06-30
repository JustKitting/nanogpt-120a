use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::mma::{
    NVFP4_PROJECTION_CTA_A_PACKS, NVFP4_PROJECTION_CTA_A_SCALES, NVFP4_PROJECTION_CTA_B_PACKS,
    NVFP4_PROJECTION_CTA_B_SCALES, Nvfp4ProjectionCtaTile, Nvfp4ProjectionParams,
    nvfp4_projection_cta_nobias_kernel_body,
    nvfp4_projection_cta_nobias_kernel_body_at_aligned_row_pair,
    nvfp4_projection_nobias_kernel_body,
};

use super::{LINEAR_BIAS_THREADS_PER_BLOCK, bias};

#[allow(static_mut_refs)]
#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn linear_backward_projection_device_scale_kernel(
        input_bytes: &[u8],
        input_scales: &[u8],
        input_global_scales: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        mut out: DisjointSlice<f32>,
        mut params: Nvfp4ProjectionParams,
    ) {
        params.weight_global_scale = weight_global_scale[0];
        nvfp4_projection_nobias_kernel_body(
            input_bytes,
            input_scales,
            input_global_scales,
            weight_bytes,
            weight_scales,
            &mut out,
            params,
        );
    }

    #[kernel]
    pub fn linear_backward_projection_cta_device_scale_kernel(
        input_bytes: &[u8],
        input_scales: &[u8],
        input_global_scales: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        mut out: DisjointSlice<f32>,
        mut params: Nvfp4ProjectionParams,
    ) {
        static mut A_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS> = SharedArray::UNINIT;
        static mut B_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS> = SharedArray::UNINIT;
        static mut A_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES> = SharedArray::UNINIT;
        static mut B_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES> = SharedArray::UNINIT;

        params.weight_global_scale = weight_global_scale[0];
        nvfp4_projection_cta_nobias_kernel_body(
            input_bytes,
            input_scales,
            input_global_scales,
            weight_bytes,
            weight_scales,
            &mut out,
            params,
            unsafe { &mut A_PACKS },
            unsafe { &mut B_PACKS },
            unsafe { &mut A_SCALES },
            unsafe { &mut B_SCALES },
        );
    }

    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    #[kernel]
    pub fn linear_backward_projection_pair_cta_device_scale_kernel(
        dinput_input_bytes: &[u8],
        dinput_input_scales: &[u8],
        dinput_input_global_scales: &[f32],
        dinput_weight_bytes: &[u8],
        dinput_weight_scales: &[u8],
        dinput_weight_global_scale: &[f32],
        mut dinput_out: DisjointSlice<f32>,
        dinput_grid_col_mask: u32,
        dinput_grid_col_shift: u32,
        dinput_tile_count: u32,
        dweight_input_bytes: &[u8],
        dweight_input_scales: &[u8],
        dweight_input_global_scales: &[f32],
        dweight_weight_bytes: &[u8],
        dweight_weight_scales: &[u8],
        dweight_weight_global_scale: &[f32],
        mut dweight_out: DisjointSlice<f32>,
        dweight_grid_col_mask: u32,
        dweight_grid_col_shift: u32,
        mut dinput_params: Nvfp4ProjectionParams,
        mut dweight_params: Nvfp4ProjectionParams,
    ) {
        static mut A_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS> = SharedArray::UNINIT;
        static mut A1_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_A_PACKS> = SharedArray::UNINIT;
        static mut B_PACKS: SharedArray<u32, NVFP4_PROJECTION_CTA_B_PACKS> = SharedArray::UNINIT;
        static mut A_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES> = SharedArray::UNINIT;
        static mut A1_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_A_SCALES> = SharedArray::UNINIT;
        static mut B_SCALES: SharedArray<u32, NVFP4_PROJECTION_CTA_B_SCALES> = SharedArray::UNINIT;

        let tile_index = thread::blockIdx_x();
        let thread_id = thread::threadIdx_x();

        if tile_index < dinput_tile_count {
            let tile_col = tile_index & dinput_grid_col_mask;
            let tile_row_pair = tile_index >> dinput_grid_col_shift;
            let tile0 =
                Nvfp4ProjectionCtaTile::from_grid_tile(tile_col, tile_row_pair * 2, thread_id);
            let tile1 =
                Nvfp4ProjectionCtaTile::from_grid_tile(tile_col, tile_row_pair * 2 + 1, thread_id);

            dinput_params.weight_global_scale = dinput_weight_global_scale[0];
            nvfp4_projection_cta_nobias_kernel_body_at_aligned_row_pair(
                dinput_input_bytes,
                dinput_input_scales,
                dinput_input_global_scales,
                dinput_weight_bytes,
                dinput_weight_scales,
                &mut dinput_out,
                dinput_params,
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
            let dweight_tile_index = tile_index - dinput_tile_count;
            let tile_col = dweight_tile_index & dweight_grid_col_mask;
            let tile_row_pair = dweight_tile_index >> dweight_grid_col_shift;
            let tile0 =
                Nvfp4ProjectionCtaTile::from_grid_tile(tile_col, tile_row_pair * 2, thread_id);
            let tile1 =
                Nvfp4ProjectionCtaTile::from_grid_tile(tile_col, tile_row_pair * 2 + 1, thread_id);

            dweight_params.weight_global_scale = dweight_weight_global_scale[0];
            nvfp4_projection_cta_nobias_kernel_body_at_aligned_row_pair(
                dweight_input_bytes,
                dweight_input_scales,
                dweight_input_global_scales,
                dweight_weight_bytes,
                dweight_weight_scales,
                &mut dweight_out,
                dweight_params,
                unsafe { &mut A_PACKS },
                unsafe { &mut A1_PACKS },
                unsafe { &mut B_PACKS },
                unsafe { &mut A_SCALES },
                unsafe { &mut A1_SCALES },
                unsafe { &mut B_SCALES },
                tile0,
                tile1,
            );
        }
    }

    #[kernel]
    pub fn linear_bias_grad_kernel(
        e: &[f32],
        mut dbias: DisjointSlice<f32>,
        token_count: u32,
        output_dim: u32,
    ) {
        static mut LOCAL_SUMS: SharedArray<f32, { LINEAR_BIAS_THREADS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;
        bias::linear_bias_grad_body(e, &mut dbias, token_count, output_dim, unsafe {
            &mut LOCAL_SUMS
        });
    }
}
