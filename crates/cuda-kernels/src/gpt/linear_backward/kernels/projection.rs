use cuda_device::{cuda_module, kernel, thread, DisjointSlice};

use crate::mma::{
    nvfp4_projection_cta_nobias_kernel_body,
    nvfp4_projection_cta_nobias_kernel_body_at_aligned_row_pair,
    nvfp4_projection_nobias_kernel_body, with_projection_cta_tiles,
    Nvfp4ProjectionCtaTile, Nvfp4ProjectionParams, ProjectionCtaAPacks, ProjectionCtaAScales,
    ProjectionCtaBPacks, ProjectionCtaBScales, ProjectionCtaRowPairTiles,
};

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn linear_backward_projection_device_scale_kernel(
        input_bytes: &[u8], input_scales: &[u8], input_global_scales: &[f32],
        weight_bytes: &[u8], weight_scales: &[u8], weight_global_scale: &[f32],
        mut out: DisjointSlice<f32>,
        mut params: Nvfp4ProjectionParams,
    ) {
        params.weight_global_scale = weight_global_scale[0];
        nvfp4_projection_nobias_kernel_body(
            input_bytes, input_scales, input_global_scales,
            weight_bytes, weight_scales, &mut out, params,
        );
    }

    #[kernel]
    pub fn linear_backward_projection_cta_device_scale_kernel(
        input_bytes: &[u8], input_scales: &[u8], input_global_scales: &[f32],
        weight_bytes: &[u8], weight_scales: &[u8], weight_global_scale: &[f32],
        mut out: DisjointSlice<f32>,
        mut params: Nvfp4ProjectionParams,
    ) {
        params.weight_global_scale = weight_global_scale[0];
        with_projection_cta_tiles!(
            nvfp4_projection_cta_nobias_kernel_body;
            input_bytes, input_scales, input_global_scales,
            weight_bytes, weight_scales, &mut out, params,
        );
    }

    #[kernel]
    pub fn linear_backward_projection_pair_cta_device_scale_kernel(
        dinput_input_bytes: &[u8], dinput_input_scales: &[u8], dinput_input_global_scales: &[f32],
        dinput_weight_bytes: &[u8], dinput_weight_scales: &[u8], dinput_weight_global_scale: &[f32],
        mut dinput_out: DisjointSlice<f32>,
        dinput_grid_col_mask: u32, dinput_grid_col_shift: u32, dinput_tile_count: u32,
        dweight_input_bytes: &[u8], dweight_input_scales: &[u8], dweight_input_global_scales: &[f32],
        dweight_weight_bytes: &[u8], dweight_weight_scales: &[u8], dweight_weight_global_scale: &[f32],
        mut dweight_out: DisjointSlice<f32>,
        dweight_grid_col_mask: u32, dweight_grid_col_shift: u32,
        mut dinput_params: Nvfp4ProjectionParams,
        mut dweight_params: Nvfp4ProjectionParams,
    ) {
        static mut A_PACKS: ProjectionCtaAPacks = ProjectionCtaAPacks::UNINIT;
        static mut A1_PACKS: ProjectionCtaAPacks = ProjectionCtaAPacks::UNINIT;
        static mut B_PACKS: ProjectionCtaBPacks = ProjectionCtaBPacks::UNINIT;
        static mut A_SCALES: ProjectionCtaAScales = ProjectionCtaAScales::UNINIT;
        static mut A1_SCALES: ProjectionCtaAScales = ProjectionCtaAScales::UNINIT;
        static mut B_SCALES: ProjectionCtaBScales = ProjectionCtaBScales::UNINIT;

        macro_rules! row_pair_tiles { () => { ProjectionCtaRowPairTiles {
            a0_packs: unsafe { &mut A_PACKS }, a1_packs: unsafe { &mut A1_PACKS }, b_packs: unsafe { &mut B_PACKS },
            a0_scales: unsafe { &mut A_SCALES }, a1_scales: unsafe { &mut A1_SCALES }, b_scales: unsafe { &mut B_SCALES },
        } }; }

        let tile_index = thread::blockIdx_x();
        let thread_id = thread::threadIdx_x();

        if tile_index < dinput_tile_count {
            let (tile0, tile1) = Nvfp4ProjectionCtaTile::packed_row_pair(
                tile_index,
                dinput_grid_col_mask,
                dinput_grid_col_shift,
                thread_id,
            );

            dinput_params.weight_global_scale = dinput_weight_global_scale[0];
            nvfp4_projection_cta_nobias_kernel_body_at_aligned_row_pair(
                dinput_input_bytes, dinput_input_scales, dinput_input_global_scales,
                dinput_weight_bytes, dinput_weight_scales, &mut dinput_out, dinput_params,
                row_pair_tiles!(), tile0, tile1,
            );
        } else {
            let dweight_tile_index = tile_index - dinput_tile_count;
            let (tile0, tile1) = Nvfp4ProjectionCtaTile::packed_row_pair(
                dweight_tile_index,
                dweight_grid_col_mask,
                dweight_grid_col_shift,
                thread_id,
            );

            dweight_params.weight_global_scale = dweight_weight_global_scale[0];
            nvfp4_projection_cta_nobias_kernel_body_at_aligned_row_pair(
                dweight_input_bytes, dweight_input_scales, dweight_input_global_scales,
                dweight_weight_bytes, dweight_weight_scales, &mut dweight_out, dweight_params,
                row_pair_tiles!(), tile0, tile1,
            );
        }
    }
}

pub(super) use module::{from_module, LoadedModule};
