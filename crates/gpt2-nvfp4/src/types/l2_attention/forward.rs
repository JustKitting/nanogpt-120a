use cuda_core::DriverError;
use rust_kernels_cuda::attention::{CProjArgs, CausalAttentionArgs, QkvProjectionArgs};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;
use rust_kernels_cuda::nvfp4_quant::{Nvfp4QuantRowwiseArgs, RowAmaxArgs};

use super::tensors::AttentionForwardArgs;
use crate::types::HiddenStateDevice;

pub(super) fn forward<'a, 'scratch>(
    args: AttentionForwardArgs<'a, 'scratch>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    let mut input_nvfp4 = args.input_nvfp4;
    let HiddenStateDevice {
        stream,
        residual,
        normalized,
        normalized_amax,
    } = args.hidden;

    args.quant_module
        .fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
            stream,
            x: normalized,
            amax: normalized_amax,
            out_fp4: &mut *input_nvfp4.bytes,
            out_scales: &mut *input_nvfp4.scales,
            out_global_scale: &mut *input_nvfp4.global_scales,
            group_count: (crate::HiddenState::LEN / 16) as u32,
            row_len: crate::GPT2_N_EMBD as u32,
        })?;

    args.module.qkv_projection(QkvProjectionArgs {
        stream,
        input: Nvfp4RowwiseDeviceTensor {
            bytes: &*input_nvfp4.bytes,
            scales: &*input_nvfp4.scales,
            global_scales: &*input_nvfp4.global_scales,
        },
        weight: args.projections.qkv_weight,
        bias: args.projections.qkv_bias,
        out: args.qkv,
        token_count: crate::GPT2_CONTEXT_LEN as u32,
        input_dim: crate::GPT2_N_EMBD as u32,
        output_dim: crate::GPT2_QKV as u32,
    })?;

    args.module.causal_attention(CausalAttentionArgs {
        stream,
        qkv: &*args.qkv,
        out: normalized,
        token_count: crate::GPT2_CONTEXT_LEN as u32,
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
    )?;

    args.module.c_proj(CProjArgs {
        stream,
        input: Nvfp4RowwiseDeviceTensor {
            bytes: &*input_nvfp4.bytes,
            scales: &*input_nvfp4.scales,
            global_scales: &*input_nvfp4.global_scales,
        },
        weight: args.projections.c_proj_weight,
        bias: args.projections.c_proj_bias,
        residual,
        token_count: crate::GPT2_CONTEXT_LEN as u32,
        embedding_dim: crate::GPT2_N_EMBD as u32,
    })?;

    Ok(HiddenStateDevice {
        stream,
        residual,
        normalized,
        normalized_amax,
    })
}

fn requantize_attention(
    quant_module: &rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule,
    stream: &cuda_core::CudaStream,
    input_nvfp4: crate::types::HiddenStateNvfp4<'_>,
    normalized: &cuda_core::DeviceBuffer<f32>,
    normalized_amax: &mut cuda_core::DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    quant_module.row_amax_f32(RowAmaxArgs {
        stream,
        x: normalized,
        out: normalized_amax,
        row_count: crate::GPT2_CONTEXT_LEN as u32,
        row_len: crate::GPT2_N_EMBD as u32,
    })?;

    quant_module.fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
        stream,
        x: normalized,
        amax: normalized_amax,
        out_fp4: &mut *input_nvfp4.bytes,
        out_scales: &mut *input_nvfp4.scales,
        out_global_scale: &mut *input_nvfp4.global_scales,
        group_count: (crate::HiddenState::LEN / 16) as u32,
        row_len: crate::GPT2_N_EMBD as u32,
    })
}
