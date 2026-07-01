use cuda_device::{DisjointSlice, cuda_module, kernel};

use crate::mma::{
    Nvfp4ProjectionParams, nvfp4_projection_cta_kernel_body, with_projection_cta_tiles,
};

#[cuda_module]
mod module {
    use super::*;

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
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
        let params = params.with_global_scales(weight_global_scale[0], bias_global_scale[0]);

        with_projection_cta_tiles!(
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

pub(crate) use module::{LoadedModule, from_module};
