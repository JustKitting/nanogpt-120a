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
) -> Result<(), DriverError> {
    let requantize_global_scale = requantize_global_scale(tensor.global_scale, step);
    optimizer.apply_adamw_update(AdamWUpdateArgs {
        stream,
        bytes: &mut tensor.bytes,
        scales: &mut tensor.scales,
        global_scale: tensor.global_scale,
        requantize_global_scale,
        grad,
        first_moment: &mut state.first,
        second_moment: &mut state.second,
        residual: &mut state.residual,
        fp32_workspace: &mut scratch.fp32_workspace,
        amax: &mut scratch.amax,
        chunk_amax: &mut scratch.chunk_amax,
        next_global_scale: &mut scratch.next_global_scale,
        len: tensor.len as u32,
        learning_rate: adam_learning_rate(step),
        weight_decay: ADAM_WEIGHT_DECAY,
        beta1: ADAM_BETA1,
        beta2: ADAM_BETA2,
        beta1_correction: 1.0 - ADAM_BETA1.powi(step as i32),
        beta2_correction: 1.0 - ADAM_BETA2.powi(step as i32),
        eps: ADAM_EPS,
    })?;

    tensor.global_scale = if requantize_global_scale > 0.0 {
        requantize_global_scale
    } else {
        scratch.next_global_scale.to_host_vec(stream)?[0]
    };
    Ok(())
}

fn requantize_global_scale(global_scale: f32, step: u32) -> f32 {
    if step == 1 { 0.0 } else { global_scale }
}
