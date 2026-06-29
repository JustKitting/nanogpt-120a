use cuda_core::{CudaStream, DriverError};

use crate::training::runtime::Runtime;
use crate::upload::UploadedBlock;

use super::super::OptimizerTrace;
use super::super::grad_block::BlockGradBuffers;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::BlockState;
use super::elapsed_ms;
use super::layer_norm::update_layer_norm;
use super::mlp::update_mlp_biases;
use super::qkv::update_qkv_biases;
use std::time::Instant;

pub(super) fn update_block(
    stream: &CudaStream,
    runtime: &Runtime,
    block: &mut UploadedBlock,
    grad: &BlockGradBuffers,
    scratch: &mut OptimizerScratch,
    state: &mut BlockState,
    step: u32,
    average_coefficient: f32,
    trace: &mut OptimizerTrace,
) -> Result<(), DriverError> {
    let optimizer = &runtime.optimizer;
    let start = Instant::now();
    update_layer_norm(
        stream,
        optimizer,
        &mut block.ln_1,
        &grad.ln_1,
        scratch,
        &mut state.ln_1,
        step,
        average_coefficient,
    )?;
    trace.adam_ms += elapsed_ms(start);

    update_qkv_biases(
        stream,
        runtime,
        block,
        grad,
        scratch,
        state,
        step,
        average_coefficient,
        trace,
    )?;

    let start = Instant::now();
    update_layer_norm(
        stream,
        optimizer,
        &mut block.ln_2,
        &grad.ln_2,
        scratch,
        &mut state.ln_2,
        step,
        average_coefficient,
    )?;
    trace.adam_ms += elapsed_ms(start);

    update_mlp_biases(
        stream,
        runtime,
        block,
        grad,
        scratch,
        state,
        step,
        average_coefficient,
        trace,
    )
}
