use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::optimizer::{AdamWUpdateArgs, OptimizerModule};

use crate::upload::UploadedNvfp4;

use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::AdamState;

const ADAM_LR: f32 = 2.0e-4;
const ADAM_WEIGHT_DECAY: f32 = 0.005;
const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.95;
const ADAM_EPS: f32 = 1.0e-10;

#[derive(Clone, Copy)]
pub(crate) struct AdamDebugConfig {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub beta1_correction: f32,
    pub beta2_correction: f32,
    pub eps: f32,
}

pub(super) fn adam_learning_rate(step: u32) -> f32 {
    ADAM_LR * super::super::learning_rate::adam_multiplier(step)
}

pub(super) fn next_latent_adam_learning_rate(step: u32) -> f32 {
    ADAM_LR * super::super::learning_rate::next_latent_adam_multiplier(step)
}

pub(crate) fn adam_debug_config(step: u32) -> AdamDebugConfig {
    AdamDebugConfig {
        learning_rate: adam_learning_rate(step),
        weight_decay: ADAM_WEIGHT_DECAY,
        beta1: ADAM_BETA1,
        beta2: ADAM_BETA2,
        beta1_correction: 1.0 - ADAM_BETA1.powi(step as i32),
        beta2_correction: 1.0 - ADAM_BETA2.powi(step as i32),
        eps: ADAM_EPS,
    }
}

pub(super) fn update_adam_tensor(
    stream: &CudaStream,
    optimizer: &OptimizerModule,
    tensor: &mut UploadedNvfp4,
    grad: &DeviceBuffer<f32>,
    scratch: &mut OptimizerScratch,
    state: &mut AdamState,
    step: u32,
    average_coefficient: f32,
) -> Result<(), DriverError> {
    update_adam_tensor_with_learning_rate(
        stream,
        optimizer,
        tensor,
        grad,
        scratch,
        state,
        step,
        average_coefficient,
        adam_learning_rate(step),
    )
}

pub(super) fn update_adam_tensor_with_learning_rate(
    stream: &CudaStream,
    optimizer: &OptimizerModule,
    tensor: &mut UploadedNvfp4,
    grad: &DeviceBuffer<f32>,
    scratch: &mut OptimizerScratch,
    state: &mut AdamState,
    step: u32,
    average_coefficient: f32,
    learning_rate: f32,
) -> Result<(), DriverError> {
    optimizer.apply_adamw_update(AdamWUpdateArgs {
        stream,
        bytes: &mut tensor.bytes,
        scales: &mut tensor.scales,
        global_scale: &mut tensor.global_scale,
        z_master: &mut state.z_master,
        x_master: &mut state.x_master,
        grad,
        first_moment: &mut state.first,
        second_moment: &mut state.second,
        amax: &mut scratch.amax,
        chunk_amax: &mut scratch.chunk_amax,
        len: tensor.len as u32,
        learning_rate,
        weight_decay: ADAM_WEIGHT_DECAY,
        beta1: ADAM_BETA1,
        beta2: ADAM_BETA2,
        beta1_correction: 1.0 - ADAM_BETA1.powi(step as i32),
        beta2_correction: 1.0 - ADAM_BETA2.powi(step as i32),
        eps: ADAM_EPS,
        average_coefficient,
    })?;

    Ok(())
}
