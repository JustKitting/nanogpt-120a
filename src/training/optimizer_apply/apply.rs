use super::super::grad_clip::GradientClipBuffers;
use super::super::grads::BackwardBuffers;
use super::super::next_latent::NextLatGradBuffers;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_aurora::{AuroraPointerTables, aurora_learning_rate};
use super::super::optimizer_state::OptimizerStateBuffers;
use super::super::optimizer_tc_scratch::AuroraScratchBuffers;
use super::super::{OptimizerTrace, TokenBatch};
use super::adam::adam_learning_rate;
use super::aurora::update_aurora_groups;
use super::base::{BaseAdamUpdateArgs, update_base_adam};
use super::block::update_block;
use super::embedding::add_embedding_lookup_grad;
use super::result::WeightUpdateResult;
use super::utils::elapsed_ms;
use crate::AppResult;
use crate::app::runtime::Runtime;
use crate::upload::UploadedModel;
use cuda_core::CudaStream;
use std::time::Instant;
pub fn apply_weight_updates(
    stream: &CudaStream,
    runtime: &Runtime,
    batch: &TokenBatch,
    uploaded: &mut UploadedModel,
    grads: &mut BackwardBuffers,
    next_latent_grads: &NextLatGradBuffers,
    scratch: &mut OptimizerScratch,
    state: &mut OptimizerStateBuffers,
    aurora: &mut AuroraScratchBuffers,
    aurora_tables: &AuroraPointerTables,
    grad_clip: &mut GradientClipBuffers,
) -> AppResult<WeightUpdateResult> {
    let optimizer = &runtime.optimizer;
    let mut trace = OptimizerTrace::default();
    let step = state.advance();
    let average_coefficient = state.schedule_free_average_coefficient(step);
    trace.adam_lr = adam_learning_rate(step);
    trace.aurora_lr = aurora_learning_rate(step);
    let start = Instant::now();
    add_embedding_lookup_grad(stream, optimizer, batch, grads, next_latent_grads)?;
    trace.embedding_lookup_ms = elapsed_ms(start);

    grad_clip.clip(stream, optimizer)?;

    let diagnostics = if super::super::diagnostics::enabled() {
        Some(
            super::super::diagnostics::PendingTrainingDiagnostics::collect(
                stream,
                uploaded,
                grads,
                state,
                step,
                average_coefficient,
            )?,
        )
    } else {
        None
    };

    let base_trace = update_base_adam(BaseAdamUpdateArgs {
        stream,
        optimizer,
        uploaded,
        grads,
        next_latent_grads,
        scratch,
        state,
        step,
        average_coefficient,
    })?;
    trace.token_embedding_ms = base_trace.token_embedding_ms;
    trace.final_norm_ms = base_trace.final_norm_ms;
    trace.adam_ms += base_trace.adam_ms;

    let start = Instant::now();
    for ((block, grad), state) in uploaded
        .blocks
        .iter_mut()
        .zip(grads.blocks.iter())
        .zip(state.blocks.iter_mut())
    {
        update_block(
            stream,
            runtime,
            block,
            grad,
            scratch,
            state,
            step,
            average_coefficient,
            &mut trace,
        )?;
    }
    trace.blocks_ms = elapsed_ms(start);

    update_aurora_groups(
        stream,
        runtime,
        aurora_tables,
        aurora,
        step,
        average_coefficient,
        &mut trace,
    )?;

    let diagnostics = diagnostics
        .map(|pending| pending.finish(stream, uploaded))
        .transpose()?;

    Ok(WeightUpdateResult { trace, diagnostics })
}
