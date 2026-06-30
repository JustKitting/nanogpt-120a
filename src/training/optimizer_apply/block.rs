use cuda_core::{CudaStream, DriverError};

use crate::training::runtime::Runtime;
use crate::upload::{UploadedBlock, UploadedModel};

use super::super::OptimizerTrace;
use super::super::grad_block::BlockGradBuffers;
use super::super::grads::BackwardBuffers;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::{BlockState, OptimizerStateBuffers};
use super::layer_norm::update_layer_norm_timed;
use super::mlp::update_mlp_biases;
use super::qkv::update_qkv_biases;
use super::timed_ms;

pub(super) fn update_blocks(
    stream: &CudaStream,
    runtime: &Runtime,
    uploaded: &mut UploadedModel,
    grads: &BackwardBuffers,
    scratch: &mut OptimizerScratch,
    state: &mut OptimizerStateBuffers,
    step: u32,
    average_coefficient: f32,
    trace: &mut OptimizerTrace,
) -> Result<(), DriverError> {
    trace.blocks_ms = timed_ms(|| {
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
                trace,
            )?;
        }
        Ok(())
    })?;
    Ok(())
}

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
    trace.adam_ms += update_layer_norm_timed(
        stream,
        optimizer,
        &mut block.ln_1,
        &grad.ln_1,
        scratch,
        &mut state.ln_1,
        step,
        average_coefficient,
    )?;

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

    trace.adam_ms += update_layer_norm_timed(
        stream,
        optimizer,
        &mut block.ln_2,
        &grad.ln_2,
        scratch,
        &mut state.ln_2,
        step,
        average_coefficient,
    )?;

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
