mod adam;
mod decode;

use super::snapshot::{PendingTensorUpdateDiagnostics, changed_bytes};
use crate::training::diagnostics::TensorUpdateDiagnostics;
use adam::adam_predicted_next;
use decode::nvfp4_host_value;

pub(super) fn tensor_update_stats(
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
        let decoded_before =
            nvfp4_host_value(&pending.before_bytes, &pending.before_scales, pending.before_global, i);
        let before = pending
            .adam
            .as_ref()
            .map(|adam| adam.x_master[i])
            .unwrap_or(decoded_before);
        let after = nvfp4_host_value(&after_bytes, &after_scales, after_global, i);
        let delta = after - before;
        let (predicted_delta, quant_error) = match pending.adam.as_ref() {
            Some(adam) => {
                let predicted_z = adam_predicted_next(adam.z_master[i], grad, adam.first[i], adam.second[i], adam);
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
    let update_to_weight_rms = ratio_or_zero(delta_rms, weight_rms_before);
    let quant_error_to_predicted_delta_rms = ratio_or_zero(quant_error_rms, predicted_delta_rms);
    let delta_grad_cos = cosine_or_zero(grad_dot_delta, grad_sum_sq, delta_sum_sq);
    let predicted_delta_grad_cos = cosine_or_zero(grad_dot_predicted_delta, grad_sum_sq, predicted_delta_sum_sq);

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

fn ratio_or_zero(numerator: f32, denominator: f32) -> f32 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

fn cosine_or_zero(dot: f64, lhs_sum_sq: f64, rhs_sum_sq: f64) -> f32 {
    if lhs_sum_sq > 0.0 && rhs_sum_sq > 0.0 {
        (dot / (lhs_sum_sq.sqrt() * rhs_sum_sq.sqrt())) as f32
    } else {
        0.0
    }
}
