use cuda_device::{DisjointSlice, cuda_module, kernel};

use crate::mma::{
    Nvfp4ProjectionParams, dispatch_projection_cta_tiles, nvfp4_projection_cta_kernel_body,
    nvfp4_projection_cta_kernel_body_at_aligned_row_pair,
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
        let params = Nvfp4ProjectionParams {
            weight_global_scale: weight_global_scale[0],
            bias_global_scale: bias_global_scale[0],
            ..params
        };

        dispatch_projection_cta_tiles!(
            params,
            nvfp4_projection_cta_kernel_body_at_aligned_row_pair,
            nvfp4_projection_cta_kernel_body;
            input_bytes,
            input_scales,
            input_global_scales,
            weight_bytes,
            weight_scales,
            bias_bytes,
            bias_scales,
            &mut out,
            params,
        );
    }
}
