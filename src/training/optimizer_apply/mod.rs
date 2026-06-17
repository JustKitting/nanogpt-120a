mod adam;
mod block;
mod layer_norm;
mod matrix;
mod mlp;
mod qkv;

use cuda_core::{CudaStream, DriverError};
use gpt2_nvfp4::GPT2_N_EMBD;
use rust_kernels_cuda::optimizer::{EmbeddingLookupGradArgs, OptimizerModule};
use std::time::Instant;

use crate::AppResult;
use crate::runtime::Runtime;
use crate::upload::UploadedModel;

use super::grads::BackwardBuffers;
use super::optimizer::OptimizerScratch;
use super::optimizer_state::OptimizerStateBuffers;
use super::optimizer_tc_scratch::AuroraScratchBuffers;
use super::{OptimizerTrace, TokenBatch};

pub(crate) use adam::adam_debug_config;
use adam::{adam_learning_rate, update_adam_tensor};
use block::update_block;
use layer_norm::update_layer_norm;

pub struct WeightUpdateResult {
    pub trace: OptimizerTrace,
    pub diagnostics: Option<super::diagnostics::TrainingDiagnostics>,
}

pub fn apply_weight_updates(
    stream: &CudaStream,
    runtime: &Runtime,
    batch: &TokenBatch,
    uploaded: &mut UploadedModel,
    grads: &mut BackwardBuffers,
    scratch: &mut OptimizerScratch,
    state: &mut OptimizerStateBuffers,
    aurora: &mut AuroraScratchBuffers,
) -> AppResult<WeightUpdateResult> {
    let optimizer = &runtime.optimizer;
    let mut trace = OptimizerTrace::default();
    let step = state.advance();
    trace.adam_lr = adam_learning_rate(step);

    let start = Instant::now();
    add_embedding_lookup_grad(stream, optimizer, batch, grads)?;
    trace.embedding_lookup_ms = elapsed_ms(start);
    let diagnostics = if super::diagnostics::enabled() {
        Some(super::diagnostics::PendingTrainingDiagnostics::collect(
            stream, uploaded, grads, state, step,
        )?)
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
            stream, runtime, block, grad, scratch, state, aurora, step, &mut trace,
        )?;
    }
    trace.blocks_ms = elapsed_ms(start);

    let diagnostics = diagnostics
        .map(|pending| pending.finish(stream, uploaded))
        .transpose()?;

    Ok(WeightUpdateResult { trace, diagnostics })
}

fn add_embedding_lookup_grad(
    stream: &CudaStream,
    optimizer: &OptimizerModule,
    batch: &TokenBatch,
    grads: &mut BackwardBuffers,
) -> Result<(), DriverError> {
    optimizer.add_embedding_lookup_grad(EmbeddingLookupGradArgs {
        stream,
        tokens: &batch.tokens,
        d_embedding_residual: &grads.d_embedding_residual,
        d_token_embedding: &mut grads.d_lm_head_weight,
        token_count: batch.token_count as u32,
        embedding_dim: GPT2_N_EMBD as u32,
    })
}

fn seed(step: u32, salt: u32) -> u32 {
    step.wrapping_mul(0x9e37_79b9) ^ salt
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}
