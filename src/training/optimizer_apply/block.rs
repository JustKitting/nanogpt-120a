use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::training::runtime::Runtime;
use crate::upload::{UploadedBlock, UploadedModel, UploadedNvfp4};

use super::super::OptimizerTrace;
use super::super::grad_block::BlockGradBuffers;
use super::super::grads::BackwardBuffers;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::{AdamState, BlockState, OptimizerStateBuffers};
use super::adam::AdamUpdate;
use super::layer_norm::update_layer_norm_timed;
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

    {
        let mut adam = AdamUpdate::new(stream, optimizer, scratch, step, average_coefficient);
        update_bias_timed(
            &mut adam,
            trace,
            &mut block.attn_qkv.bias,
            &grad.d_attn_qkv_bias,
            &mut state.attn_qkv.bias,
        )?;
        update_bias_timed(
            &mut adam,
            trace,
            &mut block.attn_c_proj.bias,
            &grad.d_attn_c_proj_bias,
            &mut state.attn_c_proj.bias,
        )?;
    }

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

    {
        let mut adam = AdamUpdate::new(stream, optimizer, scratch, step, average_coefficient);
        update_bias_timed(
            &mut adam,
            trace,
            &mut block.mlp_up.bias,
            &grad.d_mlp_c_fc_bias,
            &mut state.mlp_up.bias,
        )?;
        update_bias_timed(
            &mut adam,
            trace,
            &mut block.mlp_down.bias,
            &grad.d_mlp_c_proj_bias,
            &mut state.mlp_down.bias,
        )
    }
}

fn update_bias_timed(
    adam: &mut AdamUpdate<'_, '_>,
    trace: &mut OptimizerTrace,
    tensor: &mut UploadedNvfp4,
    grad: &DeviceBuffer<f32>,
    state: &mut AdamState,
) -> Result<(), DriverError> {
    trace.adam_ms += adam.update_timed(tensor, grad, state)?;
    Ok(())
}
