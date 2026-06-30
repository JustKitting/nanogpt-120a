use super::DebugMetricField;
use crate::training::diagnostics::{TensorUpdateDiagnostics, TrainingDiagnostics};
use crate::training::launch::CudaTrainOutput;

pub(super) fn debug_metric_value(field: DebugMetricField, item: &CudaTrainOutput) -> f64 {
    let Some(trace) = item.stats.diagnostics.as_ref() else {
        return f64::NAN;
    };

    match field {
        DebugMetricField::UpdateCount => trace.update_count as f64,
        DebugMetricField::PositiveUpdateDot => trace.positive_update_dot_count as f64,
        DebugMetricField::ZeroGradChanged => trace.zero_grad_changed_count as f64,
        DebugMetricField::MaxUpdateToWeightRms => trace.max_update_to_weight_rms as f64,
        DebugMetricField::DlogitsRms => trace.dlogits_rms as f64,
        DebugMetricField::DlogitsMax => trace.dlogits_max as f64,
        DebugMetricField::DLmHeadRms => trace.d_lm_head_rms as f64,
        DebugMetricField::DLmHeadMax => trace.d_lm_head_max as f64,
        DebugMetricField::DEmbeddingRms => trace.d_embedding_rms as f64,
        DebugMetricField::DEmbeddingMax => trace.d_embedding_max as f64,
        DebugMetricField::TokenEmbeddingGlobalBefore => trace.token_embedding_global_before as f64,
        DebugMetricField::TokenEmbeddingGlobalAfter => trace.token_embedding_global_after as f64,
        DebugMetricField::TokenEmbeddingChangedBytes => trace.token_embedding_changed_bytes as f64,
        DebugMetricField::TensorCount => trace.updates.len() as f64,
        DebugMetricField::TensorLenTotal => tensor_sum(trace, |update| update.len as f64),
        DebugMetricField::TensorGradRmsMax => tensor_max(trace, |update| update.grad_rms as f64),
        DebugMetricField::TensorGradMaxMax => tensor_max(trace, |update| update.grad_max as f64),
        DebugMetricField::TensorGradNonzeroTotal => {
            tensor_sum(trace, |update| update.grad_nonzero as f64)
        }
        DebugMetricField::TensorGradFiniteAll => tensor_all(trace, |update| update.grad_finite),
        DebugMetricField::TensorWeightRmsBeforeMax => {
            tensor_max(trace, |update| update.weight_rms_before as f64)
        }
        DebugMetricField::TensorWeightRmsAfterMax => {
            tensor_max(trace, |update| update.weight_rms_after as f64)
        }
        DebugMetricField::TensorDeltaRmsMax => tensor_max(trace, |update| update.delta_rms as f64),
        DebugMetricField::TensorDeltaMaxMax => tensor_max(trace, |update| update.delta_max as f64),
        DebugMetricField::TensorUpdateToWeightRmsMax => {
            tensor_max(trace, |update| update.update_to_weight_rms as f64)
        }
        DebugMetricField::TensorDeltaGradDotMaxAbs => {
            tensor_max_abs(trace, |update| update.delta_grad_dot as f64)
        }
        DebugMetricField::TensorDeltaGradCosMaxAbs => {
            tensor_max_abs(trace, |update| update.delta_grad_cos as f64)
        }
        DebugMetricField::TensorPredictedDeltaRmsMax => {
            tensor_max(trace, |update| update.predicted_delta_rms as f64)
        }
        DebugMetricField::TensorPredictedDeltaGradDotMaxAbs => {
            tensor_max_abs(trace, |update| update.predicted_delta_grad_dot as f64)
        }
        DebugMetricField::TensorPredictedDeltaGradCosMaxAbs => {
            tensor_max_abs(trace, |update| update.predicted_delta_grad_cos as f64)
        }
        DebugMetricField::TensorQuantErrorRmsMax => {
            tensor_max(trace, |update| update.quant_error_rms as f64)
        }
        DebugMetricField::TensorQuantErrorToPredictedDeltaRmsMax => tensor_max(trace, |update| {
            update.quant_error_to_predicted_delta_rms as f64
        }),
        DebugMetricField::TensorChangedBytesTotal => {
            tensor_sum(trace, |update| update.changed_bytes as f64)
        }
        DebugMetricField::TensorChangedScalesTotal => {
            tensor_sum(trace, |update| update.changed_scales as f64)
        }
        DebugMetricField::TensorGlobalBeforeMaxAbs => {
            tensor_max_abs(trace, |update| update.global_before as f64)
        }
        DebugMetricField::TensorGlobalAfterMaxAbs => {
            tensor_max_abs(trace, |update| update.global_after as f64)
        }
    }
}

fn tensor_sum(trace: &TrainingDiagnostics, value: impl Fn(&TensorUpdateDiagnostics) -> f64) -> f64 {
    trace.updates.iter().map(value).sum()
}

fn tensor_max(trace: &TrainingDiagnostics, value: impl Fn(&TensorUpdateDiagnostics) -> f64) -> f64 {
    max_or_nan(trace.updates.iter().map(value))
}

fn tensor_max_abs(
    trace: &TrainingDiagnostics,
    value: impl Fn(&TensorUpdateDiagnostics) -> f64,
) -> f64 {
    max_or_nan(trace.updates.iter().map(|update| value(update).abs()))
}

fn tensor_all(
    trace: &TrainingDiagnostics,
    predicate: impl Fn(&TensorUpdateDiagnostics) -> bool,
) -> f64 {
    if trace.updates.is_empty() {
        f64::NAN
    } else if trace.updates.iter().all(predicate) {
        1.0
    } else {
        0.0
    }
}

fn max_or_nan(values: impl Iterator<Item = f64>) -> f64 {
    values
        .reduce(|current, value| current.max(value))
        .unwrap_or(f64::NAN)
}
