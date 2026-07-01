use cuda_device::{DisjointSlice, cuda_module, kernel};

use super::LmHeadParams;
use crate::mma::{
    Nvfp4ProjectionParams, dispatch_projection_cta_tiles, nvfp4_projection_cta_nobias_kernel_body,
    nvfp4_projection_cta_nobias_kernel_body_at_aligned_row_pair,
};

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
        let projection =
            Nvfp4ProjectionParams::new(params.token_count, params.input_dim, params.vocab_size)
                .with_global_scales(weight_global_scale[0], 0.0);

        dispatch_projection_cta_tiles!(
            projection,
            nvfp4_projection_cta_nobias_kernel_body_at_aligned_row_pair,
            nvfp4_projection_cta_nobias_kernel_body;
            input_bytes,
            input_scales,
            input_global_scales,
            weight_bytes,
            weight_scales,
            &mut logits,
            projection,
        );
    }
}

pub(crate) use module::{LoadedModule, from_module};
