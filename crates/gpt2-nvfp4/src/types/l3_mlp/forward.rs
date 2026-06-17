use cuda_core::DriverError;
use rust_kernels_cuda::mlp::{MlpDownResidualArgs, MlpUpRelu2Args, MlpUpRelu2TapeArgs};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;
use rust_kernels_cuda::nvfp4_quant::{Nvfp4QuantRowwiseArgs, RowAmaxArgs};

use super::tensors::MlpForwardArgs;
use crate::types::HiddenStateDevice;

pub(super) fn forward<'a, 'scratch>(
    args: MlpForwardArgs<'a, 'scratch>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    let input_nvfp4 = args.scratch.input_nvfp4;
    let mut activation_nvfp4 = args.scratch.activation_nvfp4;
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

    let input = Nvfp4RowwiseDeviceTensor {
        bytes: &*input_nvfp4.bytes,
        scales: &*input_nvfp4.scales,
        global_scales: &*input_nvfp4.global_scales,
    };

    if let Some(pre_activation) = args.scratch.pre_activation {
        args.module.up_relu2_tape(MlpUpRelu2TapeArgs {
            stream,
            input,
            weight: args.projections.up.weight,
            bias: args.projections.up.bias,
            pre_activation,
            out: args.scratch.activation,
            token_count: crate::GPT2_CONTEXT_LEN as u32,
            input_dim: crate::GPT2_N_EMBD as u32,
            output_dim: crate::GPT2_MLP as u32,
        })?;
    } else {
        args.module.up_relu2(MlpUpRelu2Args {
            stream,
            input,
            weight: args.projections.up.weight,
            bias: args.projections.up.bias,
            out: args.scratch.activation,
            token_count: crate::GPT2_CONTEXT_LEN as u32,
            input_dim: crate::GPT2_N_EMBD as u32,
            output_dim: crate::GPT2_MLP as u32,
        })?;
    }

    quantize_activation(
        args.quant_module,
        stream,
        args.scratch.activation,
        activation_nvfp4.reborrow(),
        normalized_amax,
    )?;

    args.module.down_residual(MlpDownResidualArgs {
        stream,
        input: Nvfp4RowwiseDeviceTensor {
            bytes: &*activation_nvfp4.bytes,
            scales: &*activation_nvfp4.scales,
            global_scales: &*activation_nvfp4.global_scales,
        },
        weight: args.projections.down.weight,
        bias: args.projections.down.bias,
        residual,
        token_count: crate::GPT2_CONTEXT_LEN as u32,
        input_dim: crate::GPT2_MLP as u32,
        output_dim: crate::GPT2_N_EMBD as u32,
    })?;

    Ok(HiddenStateDevice {
        stream,
        residual,
        normalized,
        normalized_amax,
    })
}

fn quantize_activation(
    quant_module: &rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule,
    stream: &cuda_core::CudaStream,
    activation: &cuda_core::DeviceBuffer<f32>,
    activation_nvfp4: crate::types::MlpActivationNvfp4<'_>,
    normalized_amax: &mut cuda_core::DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    quant_module.row_amax_f32(RowAmaxArgs {
        stream,
        x: activation,
        out: normalized_amax,
        row_count: crate::GPT2_CONTEXT_LEN as u32,
        row_len: crate::GPT2_MLP as u32,
    })?;

    quant_module.fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
        stream,
        x: activation,
        amax: normalized_amax,
        out_fp4: &mut *activation_nvfp4.bytes,
        out_scales: &mut *activation_nvfp4.scales,
        out_global_scale: &mut *activation_nvfp4.global_scales,
        group_count: (crate::MlpActivation::LEN / 16) as u32,
        row_len: crate::GPT2_MLP as u32,
    })
}
