use cuda_core::DriverError;
use rust_kernels_cuda::attention::{
    ApplyRopeArgs, CProjArgs, CausalAttentionTcArgs, QkvProjectionArgs,
};

use super::tensors::AttentionForwardArgs;
use crate::types::HiddenStateDevice;

pub(super) fn forward<'a, 'scratch>(
    args: AttentionForwardArgs<'a, 'scratch>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    let mut input_nvfp4 = args.input_nvfp4;
    let mut tape = args.tape;
    let qkv_dim = crate::Gpt2Config::attention_qkv_dim(args.use_full_attention) as u32;
    let hidden = args.hidden;

    input_nvfp4.quantize_precomputed_amax(
        args.quant_module,
        hidden.stream,
        &*hidden.normalized,
        &*hidden.normalized_amax,
        hidden.row_count,
        crate::GPT2_N_EMBD as u32,
    )?;

    let input = input_nvfp4.device();
    if let Some(tape) = tape.as_mut() {
        tape.save_qkv_input(hidden.stream, input)?;
    }

    args.module.qkv_projection(QkvProjectionArgs {
        stream: hidden.stream,
        input,
        weight: args.projections.qkv_weight,
        bias: args.projections.qkv_bias,
        out: args.qkv,
        token_count: hidden.row_count,
        input_dim: crate::GPT2_N_EMBD as u32,
        output_dim: qkv_dim,
    })?;

    if args.use_full_attention {
        args.module.apply_rope(ApplyRopeArgs {
            stream: hidden.stream,
            qkv: args.qkv,
            qkv_f16: tape.as_mut().map(|tape| &mut *tape.qkv_f16),
            row_count: hidden.row_count,
            seq_len: hidden.seq_len,
            batch_size: hidden.batch_size,
            embedding_dim: crate::GPT2_N_EMBD as u32,
            qkv_dim,
            head_count: crate::GPT2_N_HEAD as u32,
            head_dim: crate::Gpt2Config::head_dim() as u32,
        })?;
    }

    let (qkv_f16, attention_out_f16) = match tape.as_mut() {
        Some(tape) if args.use_full_attention => (None, Some(&mut *tape.attention_out_f16)),
        Some(tape) => (Some(&mut *tape.qkv_f16), Some(&mut *tape.attention_out_f16)),
        None => (None, None),
    };

    let attention_args = CausalAttentionTcArgs {
        stream: hidden.stream,
        tc_module: args.tc_module,
        qkv: &*args.qkv,
        out: &mut *hidden.normalized,
        qkv_f16,
        attention_out_f16,
        log_sum_exp: args.attention_log_sum_exp,
        scratch: args.tc_scratch,
        row_count: hidden.row_count,
        seq_len: hidden.seq_len,
        batch_size: hidden.batch_size,
        embedding_dim: crate::GPT2_N_EMBD as u32,
        qkv_dim,
        head_count: crate::GPT2_N_HEAD as u32,
        head_dim: crate::Gpt2Config::head_dim() as u32,
    };
    if args.use_full_attention {
        args.module.causal_attention_tc(attention_args)?;
    } else {
        args.module.kda_attention_tc(attention_args)?;
    }

    input_nvfp4.quantize_row_amax(
        args.quant_module,
        hidden.stream,
        &*hidden.normalized,
        &mut *hidden.normalized_amax,
        hidden.row_count,
        crate::GPT2_N_EMBD as u32,
    )?;

    let input = input_nvfp4.device();
    if let Some(tape) = tape.as_mut() {
        tape.save_c_proj_input(hidden.stream, input)?;
    }

    args.module.c_proj(CProjArgs {
        stream: hidden.stream,
        input,
        weight: args.projections.c_proj_weight,
        bias: args.projections.c_proj_bias,
        residual: &mut *hidden.residual,
        token_count: hidden.row_count,
        embedding_dim: crate::GPT2_N_EMBD as u32,
    })?;

    Ok(hidden)
}
