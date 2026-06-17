use cuda_device::{DisjointSlice, cuda_module, kernel};

use crate::mma::{
    Nvfp4ProjectionParams, nvfp4_projection_kernel_body, nvfp4_projection_residual_kernel_body,
};

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
        mut out: DisjointSlice<f32>,
        params: Nvfp4ProjectionParams,
    ) {
        nvfp4_projection_kernel_body(
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

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn attention_projection_residual_tape_kernel(
        input_bytes: &[u8],
        input_scales: &[u8],
        input_global_scales: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        bias_bytes: &[u8],
        bias_scales: &[u8],
        mut residual: DisjointSlice<f32>,
        mut projection_out: DisjointSlice<f32>,
        params: Nvfp4ProjectionParams,
    ) {
        nvfp4_projection_residual_kernel_body(
            input_bytes,
            input_scales,
            input_global_scales,
            weight_bytes,
            weight_scales,
            bias_bytes,
            bias_scales,
            &mut residual,
            &mut projection_out,
            params,
        );
    }
}

pub(crate) use module::{LoadedModule, from_module};
