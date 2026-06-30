use cuda_core::DriverError;

use super::gather::TC_FORWARD_THREADS_PER_BLOCK;
use super::types::CausalAttentionTcArgs;
use crate::attention::{AttentionModule, CausalAttentionParams};
use crate::f16_tc_matmul::F16ConvertArgs;
use crate::kda_launch::{self, KDA_HEAD_DIM};
use crate::launch::{grid_x_config, linear_config};

impl AttentionModule {
    pub fn kda_attention_tc(
        &self,
        args: CausalAttentionTcArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(
            args.head_dim, KDA_HEAD_DIM,
            "KDA path currently expects head_dim=64"
        );
        let params = CausalAttentionParams::new(
            args.row_count,
            args.seq_len,
            args.batch_size,
            args.embedding_dim,
            args.qkv_dim,
            args.head_count,
            args.head_dim,
        );
        let dims = kda_launch::LaunchDims::new(
            args.batch_size,
            args.head_count,
            args.seq_len,
            args.head_dim,
            params.chunk_size,
        );
        let mm = kda_launch::MatmulRunner::new(args.stream, args.tc_module, dims.chunk_batch);
        let scratch = args.scratch;
        let kda = &self.causal_attention_tc;
        let stream = args.stream;
        let threads = TC_FORWARD_THREADS_PER_BLOCK;
        let linear = |n| linear_config(n, threads);
        let batch_cfg = grid_x_config(dims.batch_head, threads);
        let chunk_cfg = kda_launch::chunk_dim_config(dims.batch_head, dims.chunks, threads);
        let matrix_cfg = grid_x_config(dims.chunk_batch, threads);
        macro_rules! kda_linear {
            ($kernel:ident, $n:expr; $($arg:expr),* $(,)?) => {
                kda.$kernel(stream, linear($n), $($arg,)* params)?;
            };
        }
        macro_rules! kda_chunk {
            ($kernel:ident; $($arg:expr),* $(,)?) => {
                kda.$kernel(stream, chunk_cfg, $($arg,)* params)?;
            };
        }
        macro_rules! kda_batch {
            ($kernel:ident; $($arg:expr),* $(,)?) => {
                kda.$kernel(stream, batch_cfg, $($arg,)* params)?;
            };
        }
        macro_rules! kda_matrix {
            ($kernel:ident; $($arg:expr),* $(,)?) => {
                kda.$kernel(stream, matrix_cfg, $($arg,)* params)?;
            };
        }
        macro_rules! mm_in_scratch {
            ($a:ident, $b:ident, $out:ident, $shape:expr) => {
                mm.f32_input(&*scratch.$a, &*scratch.$b, &mut *scratch.$out, $shape)?;
            };
        }
        macro_rules! mm_rhs_scratch {
            ($a:ident, $b:ident, $out:ident, $shape:expr) => {
                mm.f32_rhs(&*scratch.$a, &*scratch.$b, &mut *scratch.$out, $shape)?;
            };
        }

        if let Some(qkv_f16) = args.qkv_f16 {
            args.tc_module.fp32_to_f16(F16ConvertArgs {
                stream,
                src: args.qkv,
                dst: qkv_f16,
                element_count: args.row_count * args.qkv_dim,
            })?;
        }

        kda_linear!(prepare_kda_forward_kernel, dims.batch_head * args.seq_len * 32; args.qkv, &mut *scratch.q, &mut *scratch.k, &mut *scratch.v, &mut *scratch.scores, &mut *args.log_sum_exp);
        kda_chunk!(chunk_cumsum_kda_g_kernel; &mut *scratch.scores);
        kda_linear!(make_kda_qg_kneg_kernel, dims.compact_elems; &mut *scratch.q, &*scratch.k, &*scratch.scores, &mut *scratch.compact_out);
        kda_linear!(make_kda_kg_kpos_vbeta_kernel, dims.compact_elems; &mut *scratch.k, &mut *scratch.v, &*scratch.scores, &*args.log_sum_exp, &mut *scratch.probs);
        kda_linear!(store_kda_chunk_g_last_kernel, dims.batch_head * dims.chunks * args.head_dim; &*scratch.scores, &mut *args.log_sum_exp);
        mm_in_scratch!(probs, compact_out, scores, dims.cch());
        kda_matrix!(mask_kda_akk_kernel; &mut *scratch.scores);
        kda_matrix!(solve_kda_akk_inv_kernel; &mut *scratch.scores);
        mm_rhs_scratch!(scores, probs, compact_out, dims.chc());
        mm_rhs_scratch!(scores, v, probs, dims.chc());
        kda_linear!(make_kda_kneg_from_kg_kernel, dims.compact_elems; &*scratch.k, &*args.log_sum_exp, &mut *scratch.v);
        mm_in_scratch!(q, v, scores, dims.cch());
        kda_linear!(mask_kda_aqk_kernel, dims.chunk_matrix_elems; &mut *scratch.scores);

        macro_rules! kda_output {
            ($chunk_states:expr) => {{
                kda_batch!(chunk_kda_state_save_kernel; &*scratch.k, &mut *scratch.v, &*scratch.compact_out, &*scratch.probs, &*args.log_sum_exp, &mut *$chunk_states);
                kda_chunk!(chunk_kda_output_from_state_kernel; &*scratch.q, &*scratch.v, &*scratch.scores, args.out, &*$chunk_states);
            }};
        }
        if let Some(chunk_states) = args.attention_out_f16 {
            kda_output!(chunk_states);
        } else {
            kda_output!(scratch.chunk_states);
        }
        kda.zero_kda_f32_kernel(
            stream,
            linear(dims.batch_head * args.seq_len),
            &mut *args.log_sum_exp,
            dims.batch_head * args.seq_len,
        )
    }
}
