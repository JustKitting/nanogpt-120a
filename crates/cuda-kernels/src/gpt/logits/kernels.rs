use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel};

use super::argmax::logits_argmax_body;
use super::args::{ARGMAX_WARPS_PER_BLOCK, LogitsArgmaxParams, LogitsTopKParams, TOPK_CANDIDATES};
use super::top_k::logits_top_k_body;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn logits_argmax_kernel(
        logits: &[f32],
        out_token: DisjointSlice<u32>,
        params: LogitsArgmaxParams,
    ) {
        static mut VALUES: SharedArray<f32, { ARGMAX_WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, { ARGMAX_WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;

        logits_argmax_body(logits, out_token, params, unsafe { &mut VALUES }, unsafe {
            &mut INDICES
        });
    }

    #[kernel]
    pub fn logits_top_k_kernel(
        logits: &[f32],
        out_tokens: DisjointSlice<u32>,
        out_values: DisjointSlice<f32>,
        params: LogitsTopKParams,
    ) {
        static mut VALUES: SharedArray<f32, TOPK_CANDIDATES> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, TOPK_CANDIDATES> = SharedArray::UNINIT;

        logits_top_k_body(
            logits,
            out_tokens,
            out_values,
            params,
            unsafe { &mut VALUES },
            unsafe { &mut INDICES },
        );
    }
}
