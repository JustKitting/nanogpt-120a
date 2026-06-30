use super::{AdamSnapshot, PendingTensorUpdateDiagnostics, changed_bytes};
use crate::training::diagnostics::TensorUpdateDiagnostics;

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
