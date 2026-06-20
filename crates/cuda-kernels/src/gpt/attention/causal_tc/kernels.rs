use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel};

use super::gather::gather_qkv_body;
use super::scatter::scatter_output_body;
use super::softmax::softmax_body;
use crate::attention::CausalAttentionParams;

#[allow(static_mut_refs)]
#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn gather_qkv_forward_kernel(
        qkv: &[f32],
        q: DisjointSlice<f32>,
        k: DisjointSlice<f32>,
        v: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        gather_qkv_body(qkv, q, k, v, params);
    }

    #[kernel]
    pub fn attention_softmax_forward_kernel(
        scores: &[f32],
        probs: DisjointSlice<f32>,
        log_sum_exp: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        static mut REDUCE: SharedArray<f32, 8> = SharedArray::UNINIT;
        softmax_body(scores, probs, log_sum_exp, params, unsafe { &mut REDUCE });
    }

    #[kernel]
    pub fn scatter_attention_forward_kernel(
        compact: &[f32],
        out: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        scatter_output_body(compact, out, params);
    }
}

pub(crate) use module::{LoadedModule, from_module};
