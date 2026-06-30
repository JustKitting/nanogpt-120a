use cuda_core::{CudaStream, DeviceBuffer};

use crate::AppResult;
use crate::upload::{UploadedLayerNorm, UploadedModel, UploadedNvfp4};

use super::super::grad_block::LayerNormGradBuffers;
use super::super::grads::BackwardBuffers;
use super::super::optimizer_apply::adam_debug_config;
use super::super::optimizer_state::{AdamState, LayerNormState, OptimizerStateBuffers};
use super::TensorUpdateDiagnostics;

mod stats;

use stats::tensor_update_stats;

pub(super) fn changed_bytes(before: &[u8], after: &[u8]) -> usize {
    before
        .iter()
        .zip(after.iter())
        .filter(|(before, after)| before != after)
        .count()
}

pub(super) struct PendingTensorUpdateDiagnostics {
    name: String,
    len: usize,
    before_bytes: Vec<u8>,
    before_scales: Vec<u8>,
    before_global: f32,
    grad: Vec<f32>,
    adam: Option<AdamSnapshot>,
}

struct AdamSnapshot {
    z_master: Vec<f32>,
    x_master: Vec<f32>,
    first: Vec<f32>,
    second: Vec<f32>,
    learning_rate: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    beta1_correction: f32,
    beta2_correction: f32,
    eps: f32,
    average_coefficient: f32,
}

pub(super) fn collect_update_snapshots(
    stream: &CudaStream,
    uploaded: &UploadedModel,
    grads: &BackwardBuffers,
    state: &OptimizerStateBuffers,
    step: u32,
    average_coefficient: f32,
) -> AppResult<Vec<PendingTensorUpdateDiagnostics>> {
    let mut updates = Vec::new();
    push_update(
        &mut updates,
        stream,
        "token_embedding",
        &uploaded.token_embedding,
        &grads.d_lm_head_weight,
        &state.token_embedding,
        step,
        average_coefficient,
    )?;
    push_layer_norm(
        &mut updates,
        stream,
        "ln_f",
        &uploaded.ln_f,
        &grads.final_norm,
        &state.ln_f,
        step,
        average_coefficient,
    )?;

    for (index, ((block, grad), state)) in uploaded
        .blocks
        .iter()
        .zip(grads.blocks.iter())
        .zip(state.blocks.iter())
        .enumerate()
    {
        push_layer_norm(
            &mut updates,
            stream,
            &format!("block{index}.ln_1"),
            &block.ln_1,
            &grad.ln_1,
            &state.ln_1,
            step,
            average_coefficient,
        )?;
        push_observed_update(
            &mut updates,
            stream,
            &format!("block{index}.attn_qkv.weight"),
            &block.attn_qkv.weight,
            &grad.d_attn_qkv_weight,
        )?;
        push_update(
            &mut updates,
            stream,
            &format!("block{index}.attn_qkv.bias"),
            &block.attn_qkv.bias,
            &grad.d_attn_qkv_bias,
            &state.attn_qkv.bias,
            step,
            average_coefficient,
        )?;
        push_observed_update(
            &mut updates,
            stream,
            &format!("block{index}.attn_c_proj.weight"),
            &block.attn_c_proj.weight,
            &grad.d_attn_c_proj_weight,
        )?;
        push_update(
            &mut updates,
            stream,
            &format!("block{index}.attn_c_proj.bias"),
            &block.attn_c_proj.bias,
            &grad.d_attn_c_proj_bias,
            &state.attn_c_proj.bias,
            step,
            average_coefficient,
        )?;
        push_layer_norm(
            &mut updates,
            stream,
            &format!("block{index}.ln_2"),
            &block.ln_2,
            &grad.ln_2,
            &state.ln_2,
            step,
            average_coefficient,
        )?;
        push_observed_update(
            &mut updates,
            stream,
            &format!("block{index}.mlp_up.weight"),
            &block.mlp_up.weight,
            &grad.d_mlp_c_fc_weight,
        )?;
        push_update(
            &mut updates,
            stream,
            &format!("block{index}.mlp_up.bias"),
            &block.mlp_up.bias,
            &grad.d_mlp_c_fc_bias,
            &state.mlp_up.bias,
            step,
            average_coefficient,
        )?;
        push_observed_update(
            &mut updates,
            stream,
            &format!("block{index}.mlp_down.weight"),
            &block.mlp_down.weight,
            &grad.d_mlp_c_proj_weight,
        )?;
        push_update(
            &mut updates,
            stream,
            &format!("block{index}.mlp_down.bias"),
            &block.mlp_down.bias,
            &grad.d_mlp_c_proj_bias,
            &state.mlp_down.bias,
            step,
            average_coefficient,
        )?;
    }

    Ok(updates)
}

fn push_layer_norm(
    updates: &mut Vec<PendingTensorUpdateDiagnostics>,
    stream: &CudaStream,
    name: &str,
    layer_norm: &UploadedLayerNorm,
    grads: &LayerNormGradBuffers,
    state: &LayerNormState,
    step: u32,
    average_coefficient: f32,
) -> AppResult {
    push_update(
        updates,
        stream,
        &format!("{name}.weight"),
        &layer_norm.weight,
        &grads.d_weight,
        &state.weight,
        step,
        average_coefficient,
    )?;
    push_update(
        updates,
        stream,
        &format!("{name}.bias"),
        &layer_norm.bias,
        &grads.d_bias,
        &state.bias,
        step,
        average_coefficient,
    )
}

fn push_update(
    updates: &mut Vec<PendingTensorUpdateDiagnostics>,
    stream: &CudaStream,
    name: &str,
    tensor: &UploadedNvfp4,
    grad: &DeviceBuffer<f32>,
    state: &AdamState,
    step: u32,
    average_coefficient: f32,
) -> AppResult {
    let config = adam_debug_config(step);
    let adam = AdamSnapshot {
        z_master: state.z_master.to_host_vec(stream)?,
        x_master: state.x_master.to_host_vec(stream)?,
        first: state.first.to_host_vec(stream)?,
        second: state.second.to_host_vec(stream)?,
        learning_rate: config.learning_rate,
        weight_decay: config.weight_decay,
        beta1: config.beta1,
        beta2: config.beta2,
        beta1_correction: config.beta1_correction,
        beta2_correction: config.beta2_correction,
        eps: config.eps,
        average_coefficient,
    };
    push_snapshot(updates, stream, name, tensor, grad, Some(adam))
}

fn push_observed_update(
    updates: &mut Vec<PendingTensorUpdateDiagnostics>,
    stream: &CudaStream,
    name: &str,
    tensor: &UploadedNvfp4,
    grad: &DeviceBuffer<f32>,
) -> AppResult {
    push_snapshot(updates, stream, name, tensor, grad, None)
}

fn push_snapshot(
    updates: &mut Vec<PendingTensorUpdateDiagnostics>,
    stream: &CudaStream,
    name: &str,
    tensor: &UploadedNvfp4,
    grad: &DeviceBuffer<f32>,
    adam: Option<AdamSnapshot>,
) -> AppResult {
    updates.push(PendingTensorUpdateDiagnostics {
        name: name.to_string(),
        len: tensor.len,
        before_bytes: tensor.bytes.to_host_vec(stream)?,
        before_scales: tensor.scales.to_host_vec(stream)?,
        before_global: tensor.global_scale_to_host(stream)?,
        grad: grad.to_host_vec(stream)?,
        adam,
    });
    Ok(())
}

pub(super) fn finish_update_snapshots(
    stream: &CudaStream,
    uploaded: &UploadedModel,
    pending: Vec<PendingTensorUpdateDiagnostics>,
) -> AppResult<Vec<TensorUpdateDiagnostics>> {
    let mut updates = Vec::with_capacity(pending.len());
    let mut iter = pending.into_iter();

    finish_update(
        &mut updates,
        stream,
        &uploaded.token_embedding,
        iter.next().unwrap(),
    )?;
    finish_layer_norm(&mut updates, stream, &uploaded.ln_f, &mut iter)?;

    for block in &uploaded.blocks {
        finish_layer_norm(&mut updates, stream, &block.ln_1, &mut iter)?;
        finish_update(
            &mut updates,
            stream,
            &block.attn_qkv.weight,
            iter.next().unwrap(),
        )?;
        finish_update(
            &mut updates,
            stream,
            &block.attn_qkv.bias,
            iter.next().unwrap(),
        )?;
        finish_update(
            &mut updates,
            stream,
            &block.attn_c_proj.weight,
            iter.next().unwrap(),
        )?;
        finish_update(
            &mut updates,
            stream,
            &block.attn_c_proj.bias,
            iter.next().unwrap(),
        )?;
        finish_layer_norm(&mut updates, stream, &block.ln_2, &mut iter)?;
        finish_update(
            &mut updates,
            stream,
            &block.mlp_up.weight,
            iter.next().unwrap(),
        )?;
        finish_update(
            &mut updates,
            stream,
            &block.mlp_up.bias,
            iter.next().unwrap(),
        )?;
        finish_update(
            &mut updates,
            stream,
            &block.mlp_down.weight,
            iter.next().unwrap(),
        )?;
        finish_update(
            &mut updates,
            stream,
            &block.mlp_down.bias,
            iter.next().unwrap(),
        )?;
    }

    Ok(updates)
}

fn finish_layer_norm(
    updates: &mut Vec<TensorUpdateDiagnostics>,
    stream: &CudaStream,
    layer_norm: &UploadedLayerNorm,
    iter: &mut impl Iterator<Item = PendingTensorUpdateDiagnostics>,
) -> AppResult {
    finish_update(updates, stream, &layer_norm.weight, iter.next().unwrap())?;
    finish_update(updates, stream, &layer_norm.bias, iter.next().unwrap())
}

fn finish_update(
    updates: &mut Vec<TensorUpdateDiagnostics>,
    stream: &CudaStream,
    tensor: &UploadedNvfp4,
    pending: PendingTensorUpdateDiagnostics,
) -> AppResult {
    let after_bytes = tensor.bytes.to_host_vec(stream)?;
    let after_scales = tensor.scales.to_host_vec(stream)?;
    let after_global = tensor.global_scale_to_host(stream)?;
    updates.push(tensor_update_stats(
        pending,
        after_bytes,
        after_scales,
        after_global,
    ));
    Ok(())
}
