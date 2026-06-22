use cuda_core::DriverError;
use gpt2_nvfp4::{GPT2_N_EMBD, NEXTLAT_HIDDEN, NEXTLAT_INPUT};
use rust_kernels_cuda::next_latent::{
    NextLatGeluArgs, NextLatProjectionArgs, NextLatResidualAddArgs, NextLatSmoothL1Args,
};

use super::forward::NextLatForwardArgs;
use super::quantize::{quantize_activation, rowwise};

pub(super) fn projection_gelu1(args: &mut NextLatForwardArgs<'_, '_>) -> Result<(), DriverError> {
    args.next_latent.projection(NextLatProjectionArgs {
        stream: args.stream,
        input: rowwise(
            &args.buffers.input_bytes,
            &args.buffers.input_scales,
            &args.buffers.input_globals,
        ),
        weight: args.weights.input_projection.weight.mma(),
        bias: args.weights.input_projection.bias.device(),
        out: &mut args.buffers.pre1,
        token_count: args.row_count,
        input_dim: NEXTLAT_INPUT as u32,
        output_dim: NEXTLAT_HIDDEN as u32,
    })?;
    args.next_latent.gelu(NextLatGeluArgs {
        stream: args.stream,
        input: &args.buffers.pre1,
        out: &mut args.buffers.act1,
        len: args.row_count * NEXTLAT_HIDDEN as u32,
    })?;
    let buffers = &mut args.buffers;
    quantize_activation(
        args.quant,
        args.stream,
        args.row_count,
        &buffers.act1,
        &mut buffers.normalized_amax,
        &mut buffers.act1_bytes,
        &mut buffers.act1_scales,
        &mut buffers.act1_globals,
    )
}

pub(super) fn projection_gelu2(args: &mut NextLatForwardArgs<'_, '_>) -> Result<(), DriverError> {
    args.next_latent.projection(NextLatProjectionArgs {
        stream: args.stream,
        input: rowwise(
            &args.buffers.act1_bytes,
            &args.buffers.act1_scales,
            &args.buffers.act1_globals,
        ),
        weight: args.weights.transition.weight.mma(),
        bias: args.weights.transition.bias.device(),
        out: &mut args.buffers.pre2,
        token_count: args.row_count,
        input_dim: NEXTLAT_HIDDEN as u32,
        output_dim: NEXTLAT_HIDDEN as u32,
    })?;
    args.next_latent.gelu(NextLatGeluArgs {
        stream: args.stream,
        input: &args.buffers.pre2,
        out: &mut args.buffers.act2,
        len: args.row_count * NEXTLAT_HIDDEN as u32,
    })?;
    let buffers = &mut args.buffers;
    quantize_activation(
        args.quant,
        args.stream,
        args.row_count,
        &buffers.act2,
        &mut buffers.normalized_amax,
        &mut buffers.act2_bytes,
        &mut buffers.act2_scales,
        &mut buffers.act2_globals,
    )
}

pub(super) fn output_and_loss(args: NextLatForwardArgs<'_, '_>) -> Result<(), DriverError> {
    args.next_latent.projection(NextLatProjectionArgs {
        stream: args.stream,
        input: rowwise(
            &args.buffers.act2_bytes,
            &args.buffers.act2_scales,
            &args.buffers.act2_globals,
        ),
        weight: args.weights.output_projection.weight.mma(),
        bias: args.weights.output_projection.bias.device(),
        out: &mut args.buffers.delta,
        token_count: args.row_count,
        input_dim: NEXTLAT_HIDDEN as u32,
        output_dim: GPT2_N_EMBD as u32,
    })?;
    args.next_latent.residual_add(NextLatResidualAddArgs {
        stream: args.stream,
        delta: &args.buffers.delta,
        residual: args.current_states,
        out: &mut args.buffers.predicted,
        len: args.row_count * GPT2_N_EMBD as u32,
    })?;
    args.next_latent.smooth_l1(NextLatSmoothL1Args {
        stream: args.stream,
        predicted_next_states: &args.buffers.predicted,
        target_states: args.current_states,
        losses: &mut args.buffers.losses,
        d_predicted_next_states: &mut args.buffers.d_predicted,
        batch_size: args.batch_size,
        seq_len: args.seq_len,
        embedding_dim: GPT2_N_EMBD as u32,
        lambda: args.lambda,
    })
}
