use cuda_core::DriverError;
use rust_kernels_cuda::mlp::{MlpDownResidualArgs, MlpUpRelu2Args};

use super::quantize::quantize_activation;
use super::tensors::MlpForwardArgs;
use crate::types::HiddenStateDevice;

pub(super) fn forward<'a, 'scratch>(
    args: MlpForwardArgs<'a, 'scratch>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    let mut input_nvfp4 = args.scratch.input_nvfp4;
    let mut activation_nvfp4 = args.scratch.activation_nvfp4;
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

    input_nvfp4.quantize_precomputed_amax(
        args.quant_module,
        stream,
        normalized,
        normalized_amax,
        row_count,
        crate::GPT2_N_EMBD as u32,
    )?;

    let input = input_nvfp4.device();
    if let Some(tape) = tape.as_mut() {
        tape.save_up_input(stream, input)?;
    }

    args.module.up_relu2(MlpUpRelu2Args {
        stream,
        input,
        weight: args.projections.up.weight,
        bias: args.projections.up.bias,
        pre_activation: args.scratch.pre_activation,
        out: args.scratch.activation,
        token_count: row_count,
        input_dim: crate::GPT2_N_EMBD as u32,
        output_dim: crate::GPT2_MLP as u32,
    })?;

    quantize_activation(
        args.quant_module,
        stream,
        args.scratch.activation,
        activation_nvfp4.reborrow(),
        normalized_amax,
        row_count,
    )?;

    let input = activation_nvfp4.device();
    if let Some(tape) = tape.as_mut() {
        tape.save_down_input(stream, input)?;
    }

    args.module.down_residual(MlpDownResidualArgs {
        stream,
        input,
        weight: args.projections.down.weight,
        bias: args.projections.down.bias,
        residual,
        token_count: row_count,
        input_dim: crate::GPT2_MLP as u32,
        output_dim: crate::GPT2_N_EMBD as u32,
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
