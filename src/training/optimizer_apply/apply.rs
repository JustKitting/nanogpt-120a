use super::super::grads::BackwardBuffers;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_aurora::{AuroraPointerTables, aurora_learning_rate};
use super::super::optimizer_state::OptimizerStateBuffers;
use super::super::optimizer_tc_scratch::AuroraScratchBuffers;
use super::super::{OptimizerTrace, TokenBatch};
use super::adam::{adam_learning_rate, update_adam_tensor};
use super::aurora::update_aurora_groups;
use super::block::update_block;
use super::embedding::add_embedding_lookup_grad;
use super::layer_norm::update_layer_norm;
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
    scratch: &mut OptimizerScratch,
    state: &mut OptimizerStateBuffers,
    aurora: &mut AuroraScratchBuffers,
    aurora_tables: &AuroraPointerTables,
) -> AppResult<WeightUpdateResult> {
    let optimizer = &runtime.optimizer;
    let mut trace = OptimizerTrace::default();
    let step = state.advance();
    let average_coefficient = state.schedule_free_average_coefficient(step);
    trace.adam_lr = adam_learning_rate(step);
    trace.aurora_lr = aurora_learning_rate(step);
    let start = Instant::now();
    add_embedding_lookup_grad(stream, optimizer, batch, grads)?;
    trace.embedding_lookup_ms = elapsed_ms(start);

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

    let start = Instant::now();
    update_adam_tensor(
        stream,
        optimizer,
        &mut uploaded.token_embedding,
        &grads.d_lm_head_weight,
        scratch,
        &mut state.token_embedding,
        step,
        average_coefficient,
    )?;
    trace.token_embedding_ms = elapsed_ms(start);
    trace.adam_ms += trace.token_embedding_ms;

    let start = Instant::now();
    update_layer_norm(
        stream,
        optimizer,
        &mut uploaded.ln_f,
        &grads.final_norm,
        scratch,
        &mut state.ln_f,
        step,
        average_coefficient,
    )?;
    trace.final_norm_ms = elapsed_ms(start);
    trace.adam_ms += trace.final_norm_ms;

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
