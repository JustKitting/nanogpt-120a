use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel};

use super::gather::gather_body;
use super::probs::prob_ds_body;
use super::scatter::scatter_body;
use super::softmax_d::softmax_d_f16_body;
use super::types::CausalAttentionBackwardTcParams;

#[allow(static_mut_refs)]
#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn softmax_d_f16_kernel(
        out: &[u16],
        d_out: &[f32],
        softmax_d: DisjointSlice<f32>,
        params: CausalAttentionBackwardTcParams,
    ) {
        static mut REDUCE: SharedArray<f32, 2> = SharedArray::UNINIT;
        softmax_d_f16_body(out, d_out, softmax_d, params, unsafe { &mut REDUCE });
    }

    #[kernel]
    pub fn gather_qkv_dout_kernel(
        qkv: &[u16],
        d_out_src: &[f32],
        q: DisjointSlice<u16>,
        k: DisjointSlice<u16>,
        v: DisjointSlice<u16>,
        d_out: DisjointSlice<u16>,
        params: CausalAttentionBackwardTcParams,
    ) {
        gather_body(qkv, d_out_src, q, k, v, d_out, params);
    }

    #[kernel]
    pub fn attention_prob_ds_kernel(
        scores: &[f32],
        dot: &[f32],
        log_sum_exp: &[f32],
        softmax_d: &[f32],
        p: DisjointSlice<f32>,
        ds: DisjointSlice<f32>,
        params: CausalAttentionBackwardTcParams,
    ) {
        prob_ds_body(scores, dot, log_sum_exp, softmax_d, p, ds, params);
    }

    #[kernel]
    pub fn scatter_dqkv_kernel(
        d_q: &[f32],
        d_k: &[f32],
        d_v: &[f32],
        d_qkv: DisjointSlice<f32>,
        params: CausalAttentionBackwardTcParams,
    ) {
        scatter_body(d_q, d_k, d_v, d_qkv, params);
    }
}

pub(crate) use module::{LoadedModule, from_module};
