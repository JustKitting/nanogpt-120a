use cuda_device::{DisjointSlice, cuda_module, kernel};

use super::super::kda::{
    FinishKdaGrads, add_kda_compact_body, chunk_cumsum_g_body, finish_kda_backward_body,
    gather_kda_dout_body, make_kda_kneg_from_kg_body, make_kda_kpos_from_kg_body,
    make_kda_strict_neg_matrix_body, prepare_kda_backward_inputs_body,
};
use crate::attention::CausalAttentionParams;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn prepare_kda_backward_inputs_kernel(
        qkv: &[u16], q: DisjointSlice<f32>, k: DisjointSlice<f32>,
        v: DisjointSlice<f32>, g: DisjointSlice<f32>, beta: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        prepare_kda_backward_inputs_body(qkv, q, k, v, g, beta, params);
    }

    #[kernel]
    pub fn chunk_cumsum_kda_backward_g_kernel(g: DisjointSlice<f32>, params: CausalAttentionParams) {
        chunk_cumsum_g_body(g, params);
    }

    #[kernel]
    pub fn gather_kda_dout_kernel(d_out: &[f32], compact_out: DisjointSlice<f32>, params: CausalAttentionParams) {
        gather_kda_dout_body(d_out, compact_out, params);
    }

    #[kernel]
    pub fn add_kda_compact_kernel(dst: DisjointSlice<f32>, src: &[f32], params: CausalAttentionParams) {
        add_kda_compact_body(dst, src, params);
    }

    #[kernel]
    pub fn make_kda_backward_kneg_from_kg_kernel(
        kg: &[f32], g: &[f32], kneg: DisjointSlice<f32>, params: CausalAttentionParams,
    ) {
        make_kda_kneg_from_kg_body(kg, g, kneg, params);
    }

    #[kernel]
    pub fn make_kda_backward_kpos_from_kg_kernel(
        kg: &[f32], g: &[f32], beta: &[f32], kpos: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        make_kda_kpos_from_kg_body(kg, g, beta, kpos, params);
    }

    #[kernel]
    pub fn make_kda_strict_neg_matrix_kernel(src: &[f32], dst: DisjointSlice<f32>, params: CausalAttentionParams) {
        make_kda_strict_neg_matrix_body(src, dst, params);
    }

    #[kernel]
    pub fn finish_kda_backward_kernel(
        qkv: &[u16], d_q: &[f32], d_k: &[f32], d_v: &[f32], d_g: &[f32], d_beta: &[f32],
        d_qkv: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        finish_kda_backward_body(qkv, FinishKdaGrads { q: d_q, k: d_k, v: d_v, g: d_g, beta: d_beta }, d_qkv, params);
    }
}

pub(super) use module::{LoadedModule, from_module};
