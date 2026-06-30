use cuda_core::{CudaStream, DeviceBuffer};

use crate::AppResult;
use crate::upload::{UploadedLayerNorm, UploadedModel, UploadedNvfp4};

use super::super::grad_block::LayerNormGradBuffers;
use super::super::grads::BackwardBuffers;
use super::super::optimizer_apply::adam_debug_config;
use super::super::optimizer_state::{AdamState, LayerNormState, OptimizerStateBuffers};
use super::TensorUpdateDiagnostics;

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

fn tensor_update_stats(
    pending: PendingTensorUpdateDiagnostics,
    after_bytes: Vec<u8>,
    after_scales: Vec<u8>,
    after_global: f32,
) -> TensorUpdateDiagnostics {
    let mut grad_sum_sq = 0.0f64;
    let mut weight_before_sum_sq = 0.0f64;
    let mut weight_after_sum_sq = 0.0f64;
    let mut delta_sum_sq = 0.0f64;
    let mut grad_dot_delta = 0.0f64;
    let mut predicted_delta_sum_sq = 0.0f64;
    let mut grad_dot_predicted_delta = 0.0f64;
    let mut quant_error_sum_sq = 0.0f64;
    let mut grad_max = 0.0f32;
    let mut delta_max = 0.0f32;
    let mut grad_nonzero = 0usize;
    let mut grad_finite = true;

    for i in 0..pending.len {
        let grad = pending.grad[i];
        let decoded_before = nvfp4_host_value(
            &pending.before_bytes,
            &pending.before_scales,
            pending.before_global,
            i,
        );
        let before = pending
            .adam
            .as_ref()
            .map(|adam| adam.x_master[i])
            .unwrap_or(decoded_before);
        let after = nvfp4_host_value(&after_bytes, &after_scales, after_global, i);
        let delta = after - before;
        let (predicted_delta, quant_error) = match pending.adam.as_ref() {
            Some(adam) => {
                let predicted_z = adam_predicted_next(
                    adam.z_master[i],
                    grad,
                    adam.first[i],
                    adam.second[i],
                    adam,
                );
                let predicted_x = before + adam.average_coefficient * (predicted_z - before);
                (predicted_x - before, after - predicted_x)
            }
            None => (0.0, 0.0),
        };

        grad_finite &= grad.is_finite();
        if grad != 0.0 {
            grad_nonzero += 1;
        }
        grad_max = grad_max.max(grad.abs());
        delta_max = delta_max.max(delta.abs());
        grad_sum_sq += (grad as f64) * (grad as f64);
        weight_before_sum_sq += (before as f64) * (before as f64);
        weight_after_sum_sq += (after as f64) * (after as f64);
        delta_sum_sq += (delta as f64) * (delta as f64);
        grad_dot_delta += (grad as f64) * (delta as f64);
        predicted_delta_sum_sq += (predicted_delta as f64) * (predicted_delta as f64);
        grad_dot_predicted_delta += (grad as f64) * (predicted_delta as f64);
        quant_error_sum_sq += (quant_error as f64) * (quant_error as f64);
    }

    let len = pending.len as f64;
    let grad_rms = rms(grad_sum_sq, len);
    let weight_rms_before = rms(weight_before_sum_sq, len);
    let weight_rms_after = rms(weight_after_sum_sq, len);
    let delta_rms = rms(delta_sum_sq, len);
    let predicted_delta_rms = rms(predicted_delta_sum_sq, len);
    let quant_error_rms = rms(quant_error_sum_sq, len);
    let update_to_weight_rms = if weight_rms_before > 0.0 {
        delta_rms / weight_rms_before
    } else {
        0.0
    };
    let quant_error_to_predicted_delta_rms = if predicted_delta_rms > 0.0 {
        quant_error_rms / predicted_delta_rms
    } else {
        0.0
    };
    let delta_grad_cos = if grad_sum_sq > 0.0 && delta_sum_sq > 0.0 {
        (grad_dot_delta / (grad_sum_sq.sqrt() * delta_sum_sq.sqrt())) as f32
    } else {
        0.0
    };
    let predicted_delta_grad_cos = if grad_sum_sq > 0.0 && predicted_delta_sum_sq > 0.0 {
        (grad_dot_predicted_delta / (grad_sum_sq.sqrt() * predicted_delta_sum_sq.sqrt())) as f32
    } else {
        0.0
    };

    TensorUpdateDiagnostics {
        name: pending.name,
        len: pending.len,
        grad_rms,
        grad_max,
        grad_nonzero,
        grad_finite,
        weight_rms_before,
        weight_rms_after,
        delta_rms,
        delta_max,
        update_to_weight_rms,
        delta_grad_dot: grad_dot_delta as f32,
        delta_grad_cos,
        predicted_delta_rms,
        predicted_delta_grad_dot: grad_dot_predicted_delta as f32,
        predicted_delta_grad_cos,
        quant_error_rms,
        quant_error_to_predicted_delta_rms,
        changed_bytes: changed_bytes(&pending.before_bytes, &after_bytes),
        changed_scales: changed_bytes(&pending.before_scales, &after_scales),
        global_before: pending.before_global,
        global_after: after_global,
    }
}

fn rms(sum_sq: f64, len: f64) -> f32 {
    (sum_sq / len).sqrt() as f32
}

fn adam_predicted_next(
    current: f32,
    grad: f32,
    first: f32,
    second: f32,
    config: &AdamSnapshot,
) -> f32 {
    let m = config.beta1 * first + (1.0 - config.beta1) * grad;
    let v = config.beta2 * second + (1.0 - config.beta2) * grad * grad;
    let update =
        (m / config.beta1_correction) / ((v / config.beta2_correction).sqrt() + config.eps);
    let decay = 1.0 - config.learning_rate * config.weight_decay;
    current * decay - config.learning_rate * update
}

fn nvfp4_host_value(bytes: &[u8], scales: &[u8], global_scale: f32, index: usize) -> f32 {
    let byte = bytes[index / 2];
    let payload = if index & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    };
    e2m1_host_value(payload) * e4m3_host_value(scales[index / 16]) * global_scale
}

fn e2m1_host_value(bits: u8) -> f32 {
    const VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let value = VALUES[(bits & 0x7) as usize];
    if bits & 0x8 == 0 { value } else { -value }
}

fn e4m3_host_value(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0 {
        sign * (mantissa as f32) * 2.0_f32.powi(-9)
    } else {
        sign * (1.0 + mantissa as f32 / 8.0) * 2.0_f32.powi(exponent as i32 - 7)
    }
}
