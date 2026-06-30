use super::DebugMetricField;
use crate::training::diagnostics::{TensorUpdateDiagnostics, TrainingDiagnostics};
use crate::training::launch::CudaTrainOutput;

pub(super) fn debug_metric_value(field: DebugMetricField, item: &CudaTrainOutput) -> f64 {
    let Some(trace) = item.stats.diagnostics.as_ref() else {
        return f64::NAN;
    };

    use DebugMetricField::*;
    match field {
        UpdateCount => trace.update_count as f64,
        PositiveUpdateDot => trace.positive_update_dot_count as f64,
        ZeroGradChanged => trace.zero_grad_changed_count as f64,
        MaxUpdateToWeightRms => trace.max_update_to_weight_rms as f64,
        DlogitsRms => trace.dlogits_rms as f64,
        DlogitsMax => trace.dlogits_max as f64,
        DLmHeadRms => trace.d_lm_head_rms as f64,
        DLmHeadMax => trace.d_lm_head_max as f64,
        DEmbeddingRms => trace.d_embedding_rms as f64,
        DEmbeddingMax => trace.d_embedding_max as f64,
        TokenEmbeddingGlobalBefore => trace.token_embedding_global_before as f64,
        TokenEmbeddingGlobalAfter => trace.token_embedding_global_after as f64,
        TokenEmbeddingChangedBytes => trace.token_embedding_changed_bytes as f64,
        TensorCount => trace.updates.len() as f64,
        TensorLenTotal => tensor_sum(trace, |update| update.len as f64),
        TensorGradRmsMax => tensor_max(trace, |update| update.grad_rms as f64),
        TensorGradMaxMax => tensor_max(trace, |update| update.grad_max as f64),
        TensorGradNonzeroTotal => tensor_sum(trace, |update| update.grad_nonzero as f64),
        TensorGradFiniteAll => tensor_all(trace, |update| update.grad_finite),
        TensorWeightRmsBeforeMax => tensor_max(trace, |update| update.weight_rms_before as f64),
        TensorWeightRmsAfterMax => tensor_max(trace, |update| update.weight_rms_after as f64),
        TensorDeltaRmsMax => tensor_max(trace, |update| update.delta_rms as f64),
        TensorDeltaMaxMax => tensor_max(trace, |update| update.delta_max as f64),
        TensorUpdateToWeightRmsMax => {
            tensor_max(trace, |update| update.update_to_weight_rms as f64)
        }
        TensorDeltaGradDotMaxAbs => tensor_max_abs(trace, |update| update.delta_grad_dot as f64),
        TensorDeltaGradCosMaxAbs => tensor_max_abs(trace, |update| update.delta_grad_cos as f64),
        TensorPredictedDeltaRmsMax => tensor_max(trace, |update| update.predicted_delta_rms as f64),
        TensorPredictedDeltaGradDotMaxAbs => {
            tensor_max_abs(trace, |update| update.predicted_delta_grad_dot as f64)
        }
        TensorPredictedDeltaGradCosMaxAbs => {
            tensor_max_abs(trace, |update| update.predicted_delta_grad_cos as f64)
        }
        TensorQuantErrorRmsMax => tensor_max(trace, |update| update.quant_error_rms as f64),
        TensorQuantErrorToPredictedDeltaRmsMax => tensor_max(trace, |update| {
            update.quant_error_to_predicted_delta_rms as f64
        }),
        TensorChangedBytesTotal => tensor_sum(trace, |update| update.changed_bytes as f64),
        TensorChangedScalesTotal => tensor_sum(trace, |update| update.changed_scales as f64),
        TensorGlobalBeforeMaxAbs => tensor_max_abs(trace, |update| update.global_before as f64),
        TensorGlobalAfterMaxAbs => tensor_max_abs(trace, |update| update.global_after as f64),
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
