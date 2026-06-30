use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel};

use super::gather::gather_body;
use super::kda::{
    ChunkStateMatmulMode, add_kda_compact_body, chunk_cumsum_g_body, chunk_intra_kda_backward_body,
    chunk_intra_kda_dm_body, chunk_kda_dkg_from_vnew_dh_body, chunk_state_matmul_body,
    chunkwise_kda_backward_body, finish_kda_backward_body, gather_kda_dout_body,
    make_kda_kneg_from_kg_body, make_kda_kpos_from_kg_body, make_kda_strict_neg_matrix_body,
    prepare_kda_backward_inputs_body,
};
use super::probs::prob_ds_body;
use super::scatter::scatter_body;
use super::softmax_d::softmax_d_f16_body;
use crate::attention::CausalAttentionParams;
use crate::kda_tc::{with_kda_tiles, with_tc_ab_tiles};

#[cuda_module]
pub(super) mod module {
    use super::*;

    macro_rules! call_body {
        ($func:ident; $($arg:expr),* $(,)?) => {
            $func($($arg),*);
        };
    }

    #[kernel]
    pub fn softmax_d_f16_kernel(
        out: &[u16],
        d_out: &[f32],
        softmax_d: DisjointSlice<f32>,
        params: CausalAttentionParams,
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
        params: CausalAttentionParams,
    ) {
        gather_body(qkv, d_out_src, q, k, v, d_out, params);
    }

    #[kernel]
    pub fn prepare_kda_backward_inputs_kernel(
        qkv: &[u16],
        q: DisjointSlice<f32>,
        k: DisjointSlice<f32>,
        v: DisjointSlice<f32>,
        g: DisjointSlice<f32>,
        beta: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        prepare_kda_backward_inputs_body(qkv, q, k, v, g, beta, params);
    }

    #[kernel]
    pub fn chunk_cumsum_kda_backward_g_kernel(
        g: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        chunk_cumsum_g_body(g, params);
    }

    #[kernel]
    pub fn gather_kda_dout_kernel(
        d_out: &[f32],
        compact_out: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        gather_kda_dout_body(d_out, compact_out, params);
    }

    #[kernel]
    pub fn add_kda_compact_kernel(
        dst: DisjointSlice<f32>,
        src: &[f32],
        params: CausalAttentionParams,
    ) {
        add_kda_compact_body(dst, src, params);
    }

    #[kernel]
    pub fn make_kda_backward_kneg_from_kg_kernel(
        kg: &[f32],
        g: &[f32],
        kneg: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        make_kda_kneg_from_kg_body(kg, g, kneg, params);
    }

    #[kernel]
    pub fn make_kda_backward_kpos_from_kg_kernel(
        kg: &[f32],
        g: &[f32],
        beta: &[f32],
        kpos: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        make_kda_kpos_from_kg_body(kg, g, beta, kpos, params);
    }

    #[kernel]
    pub fn make_kda_strict_neg_matrix_kernel(
        src: &[f32],
        dst: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        make_kda_strict_neg_matrix_body(src, dst, params);
    }

    #[kernel]
    pub fn chunk_kda_vnew_from_state_kernel(
        w: &[f32],
        u: &[f32],
        chunk_states: &[u16],
        v_new: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        with_tc_ab_tiles!(chunk_state_matmul_body; w, u, chunk_states, v_new, params; ChunkStateMatmulMode::VNew);
    }

    #[kernel]
    pub fn chunk_kda_dw_from_du_state_kernel(
        d_u: &[f32],
        chunk_states: &[u16],
        d_w: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        with_tc_ab_tiles!(chunk_state_matmul_body; d_u, d_u, chunk_states, d_w, params; ChunkStateMatmulMode::Dw);
    }

    #[kernel]
    pub fn chunk_kda_dqg_from_dout_state_kernel(
        d_out_compact: &[f32],
        chunk_states: &[u16],
        d_qg: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        with_tc_ab_tiles!(chunk_state_matmul_body; d_out_compact, d_out_compact, chunk_states, d_qg, params; ChunkStateMatmulMode::Dqg);
    }

    #[kernel]
    pub fn chunk_kda_dkg_from_vnew_dh_kernel(
        v_new: &[f32],
        d_h_states: &[f32],
        d_kg: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        with_tc_ab_tiles!(chunk_kda_dkg_from_vnew_dh_body; v_new, d_h_states, d_kg, params);
    }

    #[kernel]
    pub fn chunkwise_kda_backward_kernel(
        qg: &[f32],
        kg: &[f32],
        u_to_du: DisjointSlice<f32>,
        w_to_dw: DisjointSlice<f32>,
        aqk: &[f32],
        g: &[f32],
        chunk_states: &[u16],
        d_out: &[f32],
        d_h_states: DisjointSlice<f32>,
        d_aqk: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        with_kda_tiles!(backward chunkwise_kda_backward_body; qg, kg, u_to_du, w_to_dw, aqk, g, chunk_states, d_out, d_h_states, d_aqk, params);
    }

    #[kernel]
    pub fn chunk_intra_kda_dm_kernel(
        kg: &[f32],
        vbeta: &[f32],
        g: &[f32],
        beta: &[f32],
        d_u: &[f32],
        d_w: &[f32],
        d_m: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        with_tc_ab_tiles!(chunk_intra_kda_dm_body; kg, vbeta, g, beta, d_u, d_w, d_m, params);
    }

    #[kernel]
    pub fn chunk_intra_kda_backward_kernel(
        qg: &[f32],
        kg: &[f32],
        vbeta: &[f32],
        g: &[f32],
        beta: &[f32],
        d_qg_to_dv: DisjointSlice<f32>,
        d_kg: &[f32],
        d_k_a_to_dg: DisjointSlice<f32>,
        d_kpos_m: &[f32],
        d_vbeta_m: &[f32],
        d_kneg_from_b: &[f32],
        d_kpos_from_b_t: &[f32],
        d_q: DisjointSlice<f32>,
        d_k: DisjointSlice<f32>,
        d_beta: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        call_body!(chunk_intra_kda_backward_body; qg, kg, vbeta, g, beta, d_qg_to_dv, d_kg, d_k_a_to_dg, d_kpos_m, d_vbeta_m, d_kneg_from_b, d_kpos_from_b_t, d_q, d_k, d_beta, params);
    }

    #[kernel]
    pub fn finish_kda_backward_kernel(
        qkv: &[u16],
        d_q: &[f32],
        d_k: &[f32],
        d_v: &[f32],
        d_g: &[f32],
        d_beta: &[f32],
        d_qkv: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        finish_kda_backward_body(qkv, d_q, d_k, d_v, d_g, d_beta, d_qkv, params);
    }

    #[kernel]
    pub fn attention_prob_ds_kernel(
        scores: &[f32],
        dot: &[f32],
        log_sum_exp: &[f32],
        softmax_d: &[f32],
        p: DisjointSlice<f32>,
        ds: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        prob_ds_body(scores, dot, log_sum_exp, softmax_d, p, ds, params);
    }

    #[kernel]
    pub fn scatter_dqkv_kernel(
        d_q: &[f32],
        d_k: &[f32],
        d_v: &[f32],
        d_qkv: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        scatter_body(d_q, d_k, d_v, d_qkv, params);
    }
}

pub(crate) use module::{LoadedModule, from_module};
