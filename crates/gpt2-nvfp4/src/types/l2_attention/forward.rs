use cuda_core::DriverError;
use rust_kernels_cuda::attention::{CProjArgs, CausalAttentionArgs, QkvProjectionArgs};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantRowwiseArgs;

use super::quantize::requantize_attention;
use super::tensors::AttentionForwardArgs;
use crate::types::HiddenStateDevice;

pub(super) fn forward<'a, 'scratch>(
    args: AttentionForwardArgs<'a, 'scratch>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    let mut input_nvfp4 = args.input_nvfp4;
    let mut tape = args.tape;
    let HiddenStateDevice {
        stream,
        batch_size,
        seq_len,
        row_count,
        residual,
        normalized,
        normalized_amax,
        mean,
        inv_std,
    } = args.hidden;

    args.quant_module
        .fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
            stream,
            x: normalized,
            amax: normalized_amax,
            out_fp4: &mut *input_nvfp4.bytes,
            out_scales: &mut *input_nvfp4.scales,
            out_global_scale: &mut *input_nvfp4.global_scales,
            group_count: row_count * crate::GPT2_N_EMBD as u32 / 16,
            row_len: crate::GPT2_N_EMBD as u32,
        })?;

    let input = Nvfp4RowwiseDeviceTensor {
        bytes: &*input_nvfp4.bytes,
        scales: &*input_nvfp4.scales,
        global_scales: &*input_nvfp4.global_scales,
    };
    if let Some(tape) = tape.as_mut() {
        tape.save_qkv_input(stream, input)?;
    }

    args.module.qkv_projection(QkvProjectionArgs {
        stream,
        input,
        weight: args.projections.qkv_weight,
        bias: args.projections.qkv_bias,
        out: args.qkv,
        token_count: row_count,
        input_dim: crate::GPT2_N_EMBD as u32,
        output_dim: crate::GPT2_QKV as u32,
    })?;

    args.module.causal_attention(CausalAttentionArgs {
        stream,
        qkv: &*args.qkv,
        out: normalized,
        log_sum_exp: args.attention_log_sum_exp,
        row_count,
        seq_len,
        batch_size,
        embedding_dim: crate::GPT2_N_EMBD as u32,
        qkv_dim: crate::GPT2_QKV as u32,
        head_count: crate::GPT2_N_HEAD as u32,
        head_dim: (crate::GPT2_N_EMBD / crate::GPT2_N_HEAD) as u32,
    })?;

    requantize_attention(
        args.quant_module,
        stream,
        input_nvfp4.reborrow(),
        normalized,
        normalized_amax,
        row_count,
    )?;

    let input = Nvfp4RowwiseDeviceTensor {
        bytes: &*input_nvfp4.bytes,
        scales: &*input_nvfp4.scales,
        global_scales: &*input_nvfp4.global_scales,
    };
    if let Some(tape) = tape.as_mut() {
        tape.save_c_proj_input(stream, input)?;
    }

    args.module.c_proj(CProjArgs {
        stream,
        input,
        weight: args.projections.c_proj_weight,
        bias: args.projections.c_proj_bias,
        residual,
        token_count: row_count,
        embedding_dim: crate::GPT2_N_EMBD as u32,
    })?;

    Ok(HiddenStateDevice {
        stream,
        batch_size,
        seq_len,
        row_count,
        residual,
        normalized,
        normalized_amax,
        mean,
        inv_std,
    })
}
