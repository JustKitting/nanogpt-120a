use std::sync::Arc;

use burn::train::SupervisedTraining;
use burn::train::metric::{
    Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry,
    SerializedEntry,
};

use super::super::diagnostics::{TensorUpdateDiagnostics, TrainingDiagnostics};
use super::super::launch::{CudaLearningComponents, CudaTrainOutput};
use crate::training::metric_accumulator::MetricAccumulator;

#[derive(Clone, Copy)]
struct DebugMetricSpec {
    name: &'static str,
    unit: Option<&'static str>,
    higher_is_better: bool,
    field: DebugMetricField,
}

metric_fields! {
    DebugMetricField, DEBUG_METRIC_FIELDS, DebugMetricSpec, prefix "Diagnostic " {
        UpdateCount => ("update count", None, true),
        PositiveUpdateDot => ("positive update dot", None, true),
        ZeroGradChanged => ("zero grad changed", None, false),
        MaxUpdateToWeightRms => ("max update to weight RMS", None, false),
        DlogitsRms => ("dlogits RMS", None, false),
        DlogitsMax => ("dlogits max", None, false),
        DLmHeadRms => ("d lm head RMS", None, false),
        DLmHeadMax => ("d lm head max", None, false),
        DEmbeddingRms => ("d embedding RMS", None, false),
        DEmbeddingMax => ("d embedding max", None, false),
        TokenEmbeddingGlobalBefore => ("token embedding global before", None, false),
        TokenEmbeddingGlobalAfter => ("token embedding global after", None, false),
        TokenEmbeddingChangedBytes => ("token embedding changed bytes", None, true),
        TensorCount => ("tensor count", None, true),
        TensorLenTotal => ("tensor len total", None, true),
        TensorGradRmsMax => ("tensor grad RMS max", None, false),
        TensorGradMaxMax => ("tensor grad max max", None, false),
        TensorGradNonzeroTotal => ("tensor grad nonzero total", None, true),
        TensorGradFiniteAll => ("tensor grad finite all", None, true),
        TensorWeightRmsBeforeMax => ("tensor weight RMS before max", None, false),
        TensorWeightRmsAfterMax => ("tensor weight RMS after max", None, false),
        TensorDeltaRmsMax => ("tensor delta RMS max", None, false),
        TensorDeltaMaxMax => ("tensor delta max max", None, false),
        TensorUpdateToWeightRmsMax => ("tensor update to weight RMS max", None, false),
        TensorDeltaGradDotMaxAbs => ("tensor delta grad dot max abs", None, false),
        TensorDeltaGradCosMaxAbs => ("tensor delta grad cos max abs", None, false),
        TensorPredictedDeltaRmsMax => ("tensor predicted delta RMS max", None, false),
        TensorPredictedDeltaGradDotMaxAbs => ("tensor predicted delta grad dot max abs", None, false),
        TensorPredictedDeltaGradCosMaxAbs => ("tensor predicted delta grad cos max abs", None, false),
        TensorQuantErrorRmsMax => ("tensor quant error RMS max", None, false),
        TensorQuantErrorToPredictedDeltaRmsMax => ("tensor quant error to predicted delta RMS max", None, false),
        TensorChangedBytesTotal => ("tensor changed bytes total", None, true),
        TensorChangedScalesTotal => ("tensor changed scales total", None, true),
        TensorGlobalBeforeMaxAbs => ("tensor global before max abs", None, false),
        TensorGlobalAfterMaxAbs => ("tensor global after max abs", None, false),
    }
}

impl DebugMetricField {
    fn value(self, item: &CudaTrainOutput) -> f64 {
        let Some(trace) = item.stats.diagnostics.as_ref() else {
            return f64::NAN;
        };

        match self {
            Self::UpdateCount => trace.update_count as f64,
            Self::PositiveUpdateDot => trace.positive_update_dot_count as f64,
            Self::ZeroGradChanged => trace.zero_grad_changed_count as f64,
            Self::MaxUpdateToWeightRms => trace.max_update_to_weight_rms as f64,
            Self::DlogitsRms => trace.dlogits_rms as f64,
            Self::DlogitsMax => trace.dlogits_max as f64,
            Self::DLmHeadRms => trace.d_lm_head_rms as f64,
            Self::DLmHeadMax => trace.d_lm_head_max as f64,
            Self::DEmbeddingRms => trace.d_embedding_rms as f64,
            Self::DEmbeddingMax => trace.d_embedding_max as f64,
            Self::TokenEmbeddingGlobalBefore => trace.token_embedding_global_before as f64,
            Self::TokenEmbeddingGlobalAfter => trace.token_embedding_global_after as f64,
            Self::TokenEmbeddingChangedBytes => trace.token_embedding_changed_bytes as f64,
            Self::TensorCount => trace.updates.len() as f64,
            Self::TensorLenTotal => tensor_sum(trace, |update| update.len as f64),
            Self::TensorGradRmsMax => tensor_max(trace, |update| update.grad_rms as f64),
            Self::TensorGradMaxMax => tensor_max(trace, |update| update.grad_max as f64),
            Self::TensorGradNonzeroTotal => tensor_sum(trace, |update| update.grad_nonzero as f64),
            Self::TensorGradFiniteAll => tensor_all(trace, |update| update.grad_finite),
            Self::TensorWeightRmsBeforeMax => {
                tensor_max(trace, |update| update.weight_rms_before as f64)
            }
            Self::TensorWeightRmsAfterMax => {
                tensor_max(trace, |update| update.weight_rms_after as f64)
            }
            Self::TensorDeltaRmsMax => tensor_max(trace, |update| update.delta_rms as f64),
            Self::TensorDeltaMaxMax => tensor_max(trace, |update| update.delta_max as f64),
            Self::TensorUpdateToWeightRmsMax => {
                tensor_max(trace, |update| update.update_to_weight_rms as f64)
            }
            Self::TensorDeltaGradDotMaxAbs => {
                tensor_max_abs(trace, |update| update.delta_grad_dot as f64)
            }
            Self::TensorDeltaGradCosMaxAbs => {
                tensor_max_abs(trace, |update| update.delta_grad_cos as f64)
            }
            Self::TensorPredictedDeltaRmsMax => {
                tensor_max(trace, |update| update.predicted_delta_rms as f64)
            }
            Self::TensorPredictedDeltaGradDotMaxAbs => {
                tensor_max_abs(trace, |update| update.predicted_delta_grad_dot as f64)
            }
            Self::TensorPredictedDeltaGradCosMaxAbs => {
                tensor_max_abs(trace, |update| update.predicted_delta_grad_cos as f64)
            }
            Self::TensorQuantErrorRmsMax => {
                tensor_max(trace, |update| update.quant_error_rms as f64)
            }
            Self::TensorQuantErrorToPredictedDeltaRmsMax => tensor_max(trace, |update| {
                update.quant_error_to_predicted_delta_rms as f64
            }),
            Self::TensorChangedBytesTotal => {
                tensor_sum(trace, |update| update.changed_bytes as f64)
            }
            Self::TensorChangedScalesTotal => {
                tensor_sum(trace, |update| update.changed_scales as f64)
            }
            Self::TensorGlobalBeforeMaxAbs => {
                tensor_max_abs(trace, |update| update.global_before as f64)
            }
            Self::TensorGlobalAfterMaxAbs => {
                tensor_max_abs(trace, |update| update.global_after as f64)
            }
        }
    }
}

#[derive(Clone)]
struct CudaDebugMetric {
    spec: DebugMetricSpec,
    state: MetricAccumulator,
}

impl CudaDebugMetric {
    fn new(spec: DebugMetricSpec) -> Self {
        Self {
            spec,
            state: MetricAccumulator::default(),
        }
    }
}

impl Metric for CudaDebugMetric {
    type Input = CudaTrainOutput;

    fn name(&self) -> MetricName {
        Arc::new(self.spec.name.to_string())
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: self.spec.unit.map(str::to_string),
            higher_is_better: self.spec.higher_is_better,
        }
        .into()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        self.state
            .update(self.spec.field.value(item), self.spec.unit)
    }

    fn clear(&mut self) {
        self.state.clear();
    }
}

impl Numeric for CudaDebugMetric {
    fn value(&self) -> NumericEntry {
        self.state.value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

pub(super) fn register_numeric_metrics(
    mut training: SupervisedTraining<CudaLearningComponents>,
) -> SupervisedTraining<CudaLearningComponents> {
    for field in DEBUG_METRIC_FIELDS {
        training = training.metric_train_numeric(CudaDebugMetric::new(field.spec()));
    }
    training
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
