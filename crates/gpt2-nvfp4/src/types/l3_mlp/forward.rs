use cuda_core::DriverError;
use rust_kernels_cuda::mlp::{MlpDownResidualArgs, MlpUpRelu2Args};

use super::tensors::MlpForwardArgs;
use crate::types::HiddenStateDevice;

pub(super) fn forward<'a, 'scratch>(
    args: MlpForwardArgs<'a, 'scratch>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    let mut input_nvfp4 = args.scratch.input_nvfp4;
    let mut activation_nvfp4 = args.scratch.activation_nvfp4;
    let mut tape = args.tape;
    let hidden = args.hidden;

    let input = input_nvfp4.quantize_hidden_precomputed(
        args.quant_module,
        &hidden,
        crate::GPT2_EMBEDDING_DIM,
    )?;
    if let Some(tape) = tape.as_mut() {
        tape.save_up_input(hidden.stream, input)?;
    }

    args.module.up_relu2(MlpUpRelu2Args {
        stream: hidden.stream,
        input,
        weight: args.projections.up.weight,
        bias: args.projections.up.bias,
        pre_activation: args.scratch.pre_activation,
        out: args.scratch.activation,
        token_count: hidden.row_count,
        input_dim: crate::GPT2_EMBEDDING_DIM,
        output_dim: crate::GPT2_MLP_DIM,
    })?;

    activation_nvfp4.quantize_row_amax(
        args.quant_module,
        hidden.stream,
        args.scratch.activation,
        &mut *hidden.normalized_amax,
        hidden.row_count,
        crate::GPT2_MLP_DIM,
    )?;

    let input = activation_nvfp4.device();
    if let Some(tape) = tape.as_mut() {
        tape.save_down_input(hidden.stream, input)?;
    }

    args.module.down_residual(MlpDownResidualArgs {
        stream: hidden.stream,
        input,
        weight: args.projections.down.weight,
        bias: args.projections.down.bias,
        residual: &mut *hidden.residual,
        token_count: hidden.row_count,
        input_dim: crate::GPT2_MLP_DIM,
        output_dim: crate::GPT2_EMBEDDING_DIM,
    })?;

    Ok(hidden)
}
