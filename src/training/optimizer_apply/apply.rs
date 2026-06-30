use super::super::grad_clip::GradientClipBuffers;
use super::super::grads::BackwardBuffers;
use super::super::next_latent::NextLatGradBuffers;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_aurora::{AuroraPointerTables, aurora_learning_rate};
use super::super::optimizer_state::OptimizerStateBuffers;
use super::super::optimizer_tc_scratch::AuroraScratchBuffers;
use super::super::tape::ForwardTapeBuffers;
use super::super::{OptimizerTrace, TokenBatch};
use super::adam::adam_learning_rate;
use super::aurora::update_aurora_groups;
use super::base::{BaseAdamUpdateArgs, update_base_adam};
use super::block::update_blocks;
use super::embedding::add_embedding_lookup_grad;
use super::kda_clip::apply_kda_aurora_clip;
use super::result::WeightUpdateResult;
use super::skip::record_skip_decision;
use super::timed_ms;
use crate::AppResult;
use crate::training::runtime::Runtime;
use crate::upload::UploadedModel;
use cuda_core::CudaStream;

pub fn apply_weight_updates(
    stream: &CudaStream,
    runtime: &Runtime,
    batch: &TokenBatch,
    uploaded: &mut UploadedModel,
    grads: &mut BackwardBuffers,
    next_latent_grads: &NextLatGradBuffers,
    observed_loss: Option<f32>,
    scratch: &mut OptimizerScratch,
    state: &mut OptimizerStateBuffers,
    aurora: &mut AuroraScratchBuffers,
    aurora_tables: &AuroraPointerTables,
    tape: &ForwardTapeBuffers,
    grad_clip: &mut GradientClipBuffers,
) -> AppResult<WeightUpdateResult> {
    let optimizer = &runtime.optimizer;
    let mut trace = OptimizerTrace::default();
    let candidate_step = state.next_step();
    trace.adam_lr = adam_learning_rate(candidate_step);
    trace.aurora_lr = aurora_learning_rate(candidate_step);
    trace.embedding_lookup_ms =
        timed_ms(|| add_embedding_lookup_grad(stream, optimizer, batch, grads, next_latent_grads))?;

    let grad_norm = grad_clip.clip(stream, optimizer)?;
    trace.grad_norm = grad_norm;

    if record_skip_decision(
        stream,
        grads,
        next_latent_grads,
        candidate_step,
        &mut trace,
        state.should_skip_update(observed_loss, grad_norm),
    )? {
        return Ok(WeightUpdateResult {
            trace,
            diagnostics: None,
        });
    }

    let step = state.advance();
    debug_assert_eq!(step, candidate_step);
    let average_coefficient = state.schedule_free_average_coefficient(step);

    let diagnostics = super::super::diagnostics::enabled()
        .then(|| {
            super::super::diagnostics::PendingTrainingDiagnostics::collect(
                stream,
                uploaded,
                grads,
                state,
                step,
                average_coefficient,
            )
        })
        .transpose()?;

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

    update_blocks(
        stream,
        runtime,
        uploaded,
        grads,
        scratch,
        state,
        step,
        average_coefficient,
        &mut trace,
    )?;

    update_aurora_groups(
        stream,
        runtime,
        aurora_tables,
        aurora,
        step,
        average_coefficient,
        &mut trace,
    )?;
    apply_kda_aurora_clip(stream, runtime, uploaded, tape, scratch, state, &mut trace)?;

    let diagnostics = diagnostics
        .map(|pending| pending.finish(stream, uploaded))
        .transpose()?;

    Ok(WeightUpdateResult { trace, diagnostics })
}
