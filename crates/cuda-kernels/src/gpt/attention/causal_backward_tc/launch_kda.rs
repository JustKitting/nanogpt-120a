use cuda_core::DriverError;

use super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use super::types::{CausalAttentionBackwardTcArgs, CausalAttentionBackwardTcScratch};
use crate::attention::{AttentionModule, CausalAttentionParams};
use crate::kda_launch::{self, KDA_HEAD_DIM};
use crate::launch::{grid_x_config, linear_config};

impl AttentionModule {
    pub fn kda_attention_backward_tc(
        &self,
        args: CausalAttentionBackwardTcArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(
            args.head_dim, KDA_HEAD_DIM,
            "KDA path currently expects head_dim=64"
        );
        let CausalAttentionBackwardTcArgs {
            stream,
            tc_module,
            qkv,
            attention_out: chunk_states,
            d_out,
            log_sum_exp: _log_sum_exp,
            softmax_d: beta,
            d_qkv,
            scratch,
            row_count,
            seq_len,
            batch_size,
            embedding_dim,
            qkv_dim,
            head_count,
            head_dim,
        } = args;
        let params = CausalAttentionParams::new(
            row_count, seq_len, batch_size, embedding_dim, qkv_dim, head_count, head_dim,
        );
        let dims = kda_launch::LaunchDims::new(batch_size, head_count, seq_len, head_dim, params.chunk_size);
        let mm = kda_launch::MatmulRunner::new(stream, tc_module, dims.chunk_batch);
        let CausalAttentionBackwardTcScratch {
            q_f32: qg,
            k_f32: kg,
            v_f32: vbeta,
            g_f32: g,
            q: _q_half,
            k: _k_half,
            v: _v_half,
            d_out: _d_out_half,
            scores: chunk_matrix,
            dot: aqk_or_dm,
            p: local_grad,
            ds: dkg_from_state,
            d_q: kneg_vnew_dqg_dv,
            d_k: kpos_u_dw,
            d_v: w_du_dq,
            kda_d_q: dka_dg,
            kda_d_k: d_kneg_from_inverse,
            kda_d_v: dout_daqk_dvbeta,
            kda_d_g: dh_states_or_kneg,
            kda_d_beta: d_beta,
        } = scratch;
        let bwd_elementwise = &self.causal_attention_backward_tc.kda_elementwise;
        let bwd_tc = &self.causal_attention_backward_tc.kda_tc;
        let fwd = &self.causal_attention_tc;
        let threads = TC_BACKWARD_THREADS_PER_BLOCK;
        let batch_cfg = grid_x_config(dims.batch_head, threads);
        let chunk_cfg = kda_launch::chunk_dim_config(dims.batch_head, dims.chunks, threads);
        let matrix_cfg = grid_x_config(dims.chunk_batch, threads);
        macro_rules! bwd_linear {
            ($kernel:ident, $n:expr; $($arg:expr),* $(,)?) => {
                bwd_elementwise.$kernel(stream, linear_config($n, threads), $($arg,)* params)?;
            };
        }
        macro_rules! bwd_chunk {
            ($kernel:ident; $($arg:expr),* $(,)?) => {
                bwd_tc.$kernel(stream, chunk_cfg, $($arg,)* params)?;
            };
        }
        macro_rules! bwd_elementwise_chunk {
            ($kernel:ident; $($arg:expr),* $(,)?) => {
                bwd_elementwise.$kernel(stream, chunk_cfg, $($arg,)* params)?;
            };
        }
        macro_rules! bwd_batch {
            ($kernel:ident; $($arg:expr),* $(,)?) => {
                bwd_tc.$kernel(stream, batch_cfg, $($arg,)* params)?;
            };
        }
        macro_rules! fwd_linear {
            ($kernel:ident, $n:expr; $($arg:expr),* $(,)?) => {
                fwd.$kernel(stream, linear_config($n, threads), $($arg,)* params)?;
            };
        }
        macro_rules! fwd_matrix {
            ($kernel:ident; $($arg:expr),* $(,)?) => {
                fwd.$kernel(stream, matrix_cfg, $($arg,)* params)?;
            };
        }
        macro_rules! mm_in {
            ($a:expr, $b:expr, $out:expr, $shape:expr) => {
                mm.f32_input($a, $b, $out, $shape)?;
            };
        }
        macro_rules! mm_rhs {
            ($a:expr, $b:expr, $out:expr, $shape:expr) => {
                mm.f32_rhs($a, $b, $out, $shape)?;
            };
        }
        macro_rules! mm_at_rhs {
            ($a:expr, $b:expr, $out:expr, $shape:expr) => {
                mm.f32_a_transposed_rhs($a, $b, $out, $shape)?;
            };
        }

        bwd_linear!(prepare_kda_backward_inputs_kernel, dims.batch_head * seq_len * 32; qkv, qg, kg, vbeta, g, beta);
        bwd_elementwise_chunk!(chunk_cumsum_kda_backward_g_kernel; g);
        fwd_linear!(make_kda_qg_kneg_kernel, dims.compact_elems; qg, kg, g, kneg_vnew_dqg_dv);
        fwd_linear!(make_kda_kg_kpos_vbeta_kernel, dims.compact_elems; kg, vbeta, g, beta, kpos_u_dw);
        mm_in!(kpos_u_dw, kneg_vnew_dqg_dv, chunk_matrix, dims.cch());
        fwd_matrix!(mask_kda_akk_kernel; chunk_matrix);
        fwd_matrix!(solve_kda_akk_inv_kernel; chunk_matrix);
        mm_rhs!(chunk_matrix, kpos_u_dw, w_du_dq, dims.chc());
        mm_rhs!(chunk_matrix, vbeta, kpos_u_dw, dims.chc());
        mm_in!(qg, kneg_vnew_dqg_dv, aqk_or_dm, dims.cch());
        fwd_linear!(mask_kda_aqk_kernel, dims.chunk_matrix_elems; aqk_or_dm);
        bwd_chunk!(chunk_kda_vnew_from_state_kernel; w_du_dq, kpos_u_dw, chunk_states, kneg_vnew_dqg_dv);
        bwd_linear!(gather_kda_dout_kernel, dims.compact_elems; d_out, dout_daqk_dvbeta);
        mm_in!(dout_daqk_dvbeta, kneg_vnew_dqg_dv, local_grad, dims.cch());
        fwd_linear!(mask_kda_aqk_kernel, dims.chunk_matrix_elems; local_grad);
        mm_at_rhs!(aqk_or_dm, dout_daqk_dvbeta, kpos_u_dw, dims.chc());
        bwd_batch!(chunkwise_kda_backward_kernel; qg, kg, kpos_u_dw, w_du_dq, aqk_or_dm, g, chunk_states, d_out, dh_states_or_kneg, local_grad);
        bwd_chunk!(chunk_kda_dkg_from_vnew_dh_kernel; kneg_vnew_dqg_dv, dh_states_or_kneg, dkg_from_state);
        bwd_chunk!(chunk_kda_dw_from_du_state_kernel; kpos_u_dw, chunk_states, w_du_dq);
        bwd_chunk!(chunk_kda_dqg_from_dout_state_kernel; dout_daqk_dvbeta, chunk_states, kneg_vnew_dqg_dv);
        bwd_linear!(make_kda_backward_kneg_from_kg_kernel, dims.compact_elems; kg, g, dh_states_or_kneg);
        mm_in!(local_grad, dh_states_or_kneg, dout_daqk_dvbeta, dims.chc());
        bwd_linear!(add_kda_compact_kernel, dims.compact_elems; kneg_vnew_dqg_dv, dout_daqk_dvbeta);
        mm_at_rhs!(local_grad, qg, dka_dg, dims.chc());
        mm_at_rhs!(chunk_matrix, w_du_dq, dh_states_or_kneg, dims.chc());
        mm_at_rhs!(chunk_matrix, kpos_u_dw, dout_daqk_dvbeta, dims.chc());
        bwd_chunk!(chunk_intra_kda_dm_kernel; kg, vbeta, g, beta, kpos_u_dw, w_du_dq, aqk_or_dm);
        mm_in!(aqk_or_dm, chunk_matrix, local_grad, dims.ccc());
        mm_at_rhs!(chunk_matrix, local_grad, aqk_or_dm, dims.ccc());
        bwd_linear!(make_kda_strict_neg_matrix_kernel, dims.chunk_matrix_elems; aqk_or_dm, chunk_matrix);
        bwd_linear!(make_kda_backward_kneg_from_kg_kernel, dims.compact_elems; kg, g, kpos_u_dw);
        mm_rhs!(chunk_matrix, kpos_u_dw, d_kneg_from_inverse, dims.chc());
        bwd_linear!(make_kda_backward_kpos_from_kg_kernel, dims.compact_elems; kg, g, beta, kpos_u_dw);
        mm_at_rhs!(chunk_matrix, kpos_u_dw, local_grad, dims.chc());
        bwd_chunk!(chunk_intra_kda_backward_kernel; qg, kg, vbeta, g, beta, kneg_vnew_dqg_dv, dkg_from_state, dka_dg, dh_states_or_kneg, dout_daqk_dvbeta, d_kneg_from_inverse, local_grad, w_du_dq, chunk_matrix, d_beta);
        // After chunk_intra_kda_backward_kernel these reused buffers hold final compact gradients.
        bwd_linear!(finish_kda_backward_kernel, dims.batch_head * seq_len * 32; qkv, w_du_dq, chunk_matrix, kneg_vnew_dqg_dv, dka_dg, d_beta, d_qkv);
        Ok(())
    }
}
