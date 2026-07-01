mod accumulator;
mod adam;
mod decode;

use super::snapshot::{PendingTensorUpdateDiagnostics, changed_bytes};
use crate::training::diagnostics::TensorUpdateDiagnostics;
use accumulator::{UpdateStatAccumulator, cosine_or_zero, ratio_or_zero, rms};
use adam::adam_predicted_next;
use decode::nvfp4_host_value;

pub(super) fn tensor_update_stats(
    pending: PendingTensorUpdateDiagnostics,
    after_bytes: Vec<u8>,
    after_scales: Vec<u8>,
    after_global: f32,
) -> TensorUpdateDiagnostics {
    let mut totals = UpdateStatAccumulator::new();

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

        totals.record(grad, before, after, delta, predicted_delta, quant_error);
    }

    let len = pending.len as f64;
    let grad_rms = rms(totals.grad_sum_sq, len);
    let weight_rms_before = rms(totals.weight_before_sum_sq, len);
    let weight_rms_after = rms(totals.weight_after_sum_sq, len);
    let delta_rms = rms(totals.delta_sum_sq, len);
    let predicted_delta_rms = rms(totals.predicted_delta_sum_sq, len);
    let quant_error_rms = rms(totals.quant_error_sum_sq, len);
    let update_to_weight_rms = ratio_or_zero(delta_rms, weight_rms_before);
    let quant_error_to_predicted_delta_rms = ratio_or_zero(quant_error_rms, predicted_delta_rms);
    let delta_grad_cos = cosine_or_zero(
        totals.grad_dot_delta,
        totals.grad_sum_sq,
        totals.delta_sum_sq,
    );
    let predicted_delta_grad_cos = cosine_or_zero(
        totals.grad_dot_predicted_delta,
        totals.grad_sum_sq,
        totals.predicted_delta_sum_sq,
    );

    TensorUpdateDiagnostics {
        name: pending.name,
        len: pending.len,
        grad_rms,
        grad_max: totals.grad_max,
        grad_nonzero: totals.grad_nonzero,
        grad_finite: totals.grad_finite,
        weight_rms_before,
        weight_rms_after,
        delta_rms,
        delta_max: totals.delta_max,
        update_to_weight_rms,
        delta_grad_dot: totals.grad_dot_delta as f32,
        delta_grad_cos,
        predicted_delta_rms,
        predicted_delta_grad_dot: totals.grad_dot_predicted_delta as f32,
        predicted_delta_grad_cos,
        quant_error_rms,
        quant_error_to_predicted_delta_rms,
        changed_bytes: changed_bytes(&pending.before_bytes, &after_bytes),
        changed_scales: changed_bytes(&pending.before_scales, &after_scales),
        global_before: pending.before_global,
        global_after: after_global,
    }
}
