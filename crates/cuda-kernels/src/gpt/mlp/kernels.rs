use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::f16_tc_matmul::convert::cvt_f32_f16;
use crate::float_ptx::max_f32;
use crate::mma::{
    Nvfp4ProjectionParams, dispatch_projection_cta_tiles, nvfp4_projection_cta_kernel_body,
    nvfp4_projection_cta_kernel_body_at_aligned_row_pair, nvfp4_projection_cta_relu2_kernel_body,
    nvfp4_projection_cta_relu2_kernel_body_at_aligned_row_pair,
};

pub(super) const RELU2_THREADS_PER_BLOCK: u32 = 256;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[cuda_module]
mod module {
    use super::*;

    #[kernel]
    pub fn mlp_projection_kernel(
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

        dispatch_projection_cta_tiles!(
            params,
            nvfp4_projection_cta_kernel_body_at_aligned_row_pair,
            nvfp4_projection_cta_kernel_body;
            input_bytes, input_scales, input_global_scales,
            weight_bytes, weight_scales, bias_bytes, bias_scales,
            &mut out, params,
        );
    }

    #[kernel]
    pub fn mlp_projection_relu2_kernel(
        input_bytes: &[u8],
        input_scales: &[u8],
        input_global_scales: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        bias_bytes: &[u8],
        bias_scales: &[u8],
        weight_global_scale: &[f32],
        bias_global_scale: &[f32],
        mut pre_activation: DisjointSlice<f32>,
        mut out: DisjointSlice<f32>,
        params: Nvfp4ProjectionParams,
    ) {
        let params = params.with_global_scales(weight_global_scale[0], bias_global_scale[0]);

        dispatch_projection_cta_tiles!(
            params,
            nvfp4_projection_cta_relu2_kernel_body_at_aligned_row_pair,
            nvfp4_projection_cta_relu2_kernel_body;
            input_bytes, input_scales, input_global_scales,
            weight_bytes, weight_scales, bias_bytes, bias_scales,
            &mut pre_activation, &mut out, params,
        );
    }

    #[kernel]
    pub fn relu2_backward_kernel(
        pre_activation: &[f32],
        d_out: &[f32],
        mut d_pre_activation: DisjointSlice<f32>,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * super::RELU2_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            let relu = max_f32(pre_activation[index as usize], 0.0);
            unsafe {
                *d_pre_activation.get_unchecked_mut(index as usize) =
                    d_out[index as usize] * 2.0 * relu;
            }
        }
    }

    #[kernel]
    pub fn relu2_backward_f16_kernel(
        pre_activation: &[u16],
        d_out: &[f32],
        mut d_pre_activation: DisjointSlice<f32>,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * super::RELU2_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            let pre = cvt_f32_f16(pre_activation[index as usize]);
            let relu = max_f32(pre, 0.0);
            unsafe {
                *d_pre_activation.get_unchecked_mut(index as usize) =
                    d_out[index as usize] * 2.0 * relu;
            }
        }
    }
}

pub(super) use module::{LoadedModule, from_module};
