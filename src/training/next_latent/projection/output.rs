use cuda_core::DriverError;
use gpt2_nvfp4::{GPT2_EMBEDDING_DIM, NEXTLAT_HIDDEN_DIM};
use rust_kernels_cuda::next_latent::{
    NextLatProjectionArgs, NextLatResidualAddArgs, NextLatSmoothL1Args,
};

use super::super::forward::NextLatForwardArgs;

pub(in crate::training::next_latent) fn output_and_loss(
    args: NextLatForwardArgs<'_, '_>,
) -> Result<(), DriverError> {
    args.next_latent.projection(NextLatProjectionArgs {
        stream: args.stream,
        input: args.buffers.act2_quant.rowwise(),
        weight: args.weights.output_projection.weight.mma(),
        bias: args.weights.output_projection.bias.device(),
        out: &mut args.buffers.delta,
        token_count: args.row_count,
        input_dim: NEXTLAT_HIDDEN_DIM,
        output_dim: GPT2_EMBEDDING_DIM,
    })?;
    args.next_latent.residual_add(NextLatResidualAddArgs {
        stream: args.stream,
        delta: &args.buffers.delta,
        residual: args.current_states,
        out: &mut args.buffers.predicted,
        len: args.row_count * GPT2_EMBEDDING_DIM,
    })?;
    args.next_latent.smooth_l1(NextLatSmoothL1Args {
        stream: args.stream,
        predicted_next_states: &args.buffers.predicted,
        target_states: args.current_states,
        losses: &mut args.buffers.losses,
        d_predicted_next_states: &mut args.buffers.d_predicted,
        batch_size: args.batch_size,
        seq_len: args.seq_len,
        embedding_dim: GPT2_EMBEDDING_DIM,
        lambda: args.lambda,
    })
}
