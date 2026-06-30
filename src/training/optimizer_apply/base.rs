use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::optimizer::OptimizerModule;

use crate::training::grads::BackwardBuffers;
use crate::training::next_latent::NextLatGradBuffers;
use crate::training::optimizer::OptimizerScratch;
use crate::training::optimizer_state::OptimizerStateBuffers;
use crate::upload::UploadedModel;

use super::adam::AdamUpdate;
use super::layer_norm::update_layer_norm_timed;
use super::next_latent::{NextLatUpdateArgs, update_next_latent};
use super::timed_ms;

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
    let (token_embedding_ms, final_norm_ms) = {
        let mut adam = AdamUpdate::new(
            args.stream,
            args.optimizer,
            args.scratch,
            args.step,
            args.average_coefficient,
        );
        let token_embedding_ms = adam.update_timed(
            &mut args.uploaded.token_embedding,
            &args.grads.d_lm_head_weight,
            &mut args.state.token_embedding,
        )?;
        let final_norm_ms = update_layer_norm_timed(
            &mut adam,
            &mut args.uploaded.ln_f,
            &args.grads.final_norm,
            &mut args.state.ln_f,
        )?;
        (token_embedding_ms, final_norm_ms)
    };

    let next_latent_ms = timed_ms(|| {
        update_next_latent(NextLatUpdateArgs {
            stream: args.stream,
            optimizer: args.optimizer,
            weights: &mut args.uploaded.next_latent,
            grads: args.next_latent_grads,
            scratch: args.scratch,
            state: &mut args.state.next_latent,
            step: args.step,
            average_coefficient: args.average_coefficient,
        })
    })?;

    Ok(BaseAdamTrace {
        token_embedding_ms,
        final_norm_ms,
        adam_ms: token_embedding_ms + final_norm_ms + next_latent_ms,
    })
}
