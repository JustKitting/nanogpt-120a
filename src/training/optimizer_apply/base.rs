use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::optimizer::OptimizerModule;
use std::time::Instant;

use crate::training::grads::BackwardBuffers;
use crate::training::next_latent::NextLatGradBuffers;
use crate::training::optimizer::OptimizerScratch;
use crate::training::optimizer_state::OptimizerStateBuffers;
use crate::upload::UploadedModel;

use super::adam::update_adam_tensor;
use super::layer_norm::update_layer_norm;
use super::next_latent::{NextLatUpdateArgs, update_next_latent};
use super::utils::elapsed_ms;

pub(super) struct BaseAdamUpdateArgs<'a> {
    pub stream: &'a CudaStream,
    pub optimizer: &'a OptimizerModule,
    pub uploaded: &'a mut UploadedModel,
    pub grads: &'a BackwardBuffers,
    pub next_latent_grads: &'a NextLatGradBuffers,
    pub scratch: &'a mut OptimizerScratch,
    pub state: &'a mut OptimizerStateBuffers,
    pub step: u32,
    pub average_coefficient: f32,
}

pub(super) struct BaseAdamTrace {
    pub token_embedding_ms: f64,
    pub final_norm_ms: f64,
    pub adam_ms: f64,
}

pub(super) fn update_base_adam(args: BaseAdamUpdateArgs<'_>) -> Result<BaseAdamTrace, DriverError> {
    let token_start = Instant::now();
    update_adam_tensor(
        args.stream,
        args.optimizer,
        &mut args.uploaded.token_embedding,
        &args.grads.d_lm_head_weight,
        args.scratch,
        &mut args.state.token_embedding,
        args.step,
        args.average_coefficient,
    )?;
    let token_embedding_ms = elapsed_ms(token_start);

    let final_start = Instant::now();
    update_layer_norm(
        args.stream,
        args.optimizer,
        &mut args.uploaded.ln_f,
        &args.grads.final_norm,
        args.scratch,
        &mut args.state.ln_f,
        args.step,
        args.average_coefficient,
    )?;
    let final_norm_ms = elapsed_ms(final_start);

    let next_start = Instant::now();
    update_next_latent(NextLatUpdateArgs {
        stream: args.stream,
        optimizer: args.optimizer,
        weights: &mut args.uploaded.next_latent,
        grads: args.next_latent_grads,
        scratch: args.scratch,
        state: &mut args.state.next_latent,
        step: args.step,
        average_coefficient: args.average_coefficient,
    })?;

    Ok(BaseAdamTrace {
        token_embedding_ms,
        final_norm_ms,
        adam_ms: token_embedding_ms + final_norm_ms + elapsed_ms(next_start),
    })
}
