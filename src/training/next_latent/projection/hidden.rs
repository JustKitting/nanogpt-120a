use cuda_core::DriverError;
use gpt2_nvfp4::{NEXTLAT_HIDDEN_DIM, NEXTLAT_INPUT_DIM};
use rust_kernels_cuda::next_latent::{NextLatGeluArgs, NextLatProjectionArgs};

use super::super::forward::NextLatForwardArgs;
use super::super::quantize::quantize_activation;

pub(in crate::training::next_latent) fn projection_gelu1(
    args: &mut NextLatForwardArgs<'_, '_>,
) -> Result<(), DriverError> {
    args.next_latent.projection(NextLatProjectionArgs {
        stream: args.stream,
        input: args.buffers.input_quant.rowwise(),
        weight: args.weights.input_projection.weight.mma(),
        bias: args.weights.input_projection.bias.device(),
        out: &mut args.buffers.pre1,
        token_count: args.row_count,
        input_dim: NEXTLAT_INPUT_DIM,
        output_dim: NEXTLAT_HIDDEN_DIM,
    })?;
    args.next_latent.gelu(NextLatGeluArgs {
        stream: args.stream,
        input: &args.buffers.pre1,
        out: &mut args.buffers.act1,
        len: args.row_count * NEXTLAT_HIDDEN_DIM,
    })?;
    let buffers = &mut args.buffers;
    quantize_activation(
        args.quant,
        args.stream,
        args.row_count,
        buffers.act1_quantize(),
    )
}

pub(in crate::training::next_latent) fn projection_gelu2(
    args: &mut NextLatForwardArgs<'_, '_>,
) -> Result<(), DriverError> {
    args.next_latent.projection(NextLatProjectionArgs {
        stream: args.stream,
        input: args.buffers.act1_quant.rowwise(),
        weight: args.weights.transition.weight.mma(),
        bias: args.weights.transition.bias.device(),
        out: &mut args.buffers.pre2,
        token_count: args.row_count,
        input_dim: NEXTLAT_HIDDEN_DIM,
        output_dim: NEXTLAT_HIDDEN_DIM,
    })?;
    args.next_latent.gelu(NextLatGeluArgs {
        stream: args.stream,
        input: &args.buffers.pre2,
        out: &mut args.buffers.act2,
        len: args.row_count * NEXTLAT_HIDDEN_DIM,
    })?;
    let buffers = &mut args.buffers;
    quantize_activation(
        args.quant,
        args.stream,
        args.row_count,
        buffers.act2_quantize(),
    )
}
