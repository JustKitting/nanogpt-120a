use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;

use burn::train::SupervisedTraining;
use burn::train::metric::{
    Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry,
    SerializedEntry,
};

use super::diagnostics::{self, TensorUpdateDiagnostics, TrainingDiagnostics};
use super::launch::{CudaLearningComponents, CudaTrainOutput};
use crate::AppResult;

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

#[derive(Clone, Copy)]
struct DebugTextMetricSpec {
    name: &'static str,
    field: DebugTextField,
}

#[derive(Clone, Copy)]
enum DebugTextField {
    TokenEmbeddingHashBefore,
    TokenEmbeddingHashAfter,
    TensorNames,
}

const DEBUG_TEXT_FIELDS: &[DebugTextField] = &[
    DebugTextField::TokenEmbeddingHashBefore,
    DebugTextField::TokenEmbeddingHashAfter,
    DebugTextField::TensorNames,
];

impl DebugTextField {
    const fn spec(self) -> DebugTextMetricSpec {
        match self {
            Self::TokenEmbeddingHashBefore => DebugTextMetricSpec {
                name: "Diagnostic token embedding hash before",
                field: self,
            },
            Self::TokenEmbeddingHashAfter => DebugTextMetricSpec {
                name: "Diagnostic token embedding hash after",
                field: self,
            },
            Self::TensorNames => DebugTextMetricSpec {
                name: "Diagnostic tensor names",
                field: self,
            },
        }
    }

    fn value(self, item: &CudaTrainOutput) -> String {
        let Some(trace) = item.stats.diagnostics.as_ref() else {
            return String::new();
        };

        match self {
            Self::TokenEmbeddingHashBefore => {
                format!("{:016x}", trace.token_embedding_hash_before)
            }
            Self::TokenEmbeddingHashAfter => {
                format!("{:016x}", trace.token_embedding_hash_after)
            }
            Self::TensorNames => trace
                .updates
                .iter()
                .map(|update| update.name.as_str())
                .collect::<Vec<_>>()
                .join(","),
        }
    }
}

#[derive(Clone)]
struct CudaDebugMetric {
    spec: DebugMetricSpec,
    state: DebugMetricAccumulator,
}

impl CudaDebugMetric {
    fn new(spec: DebugMetricSpec) -> Self {
        Self {
            spec,
            state: DebugMetricAccumulator::default(),
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

#[derive(Clone)]
struct CudaDebugTextMetric {
    spec: DebugTextMetricSpec,
}

impl CudaDebugTextMetric {
    fn new(spec: DebugTextMetricSpec) -> Self {
        Self { spec }
    }
}

impl Metric for CudaDebugTextMetric {
    type Input = CudaTrainOutput;

    fn name(&self) -> MetricName {
        Arc::new(self.spec.name.to_string())
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let value = self.spec.field.value(item);
        SerializedEntry {
            formatted: value.clone(),
            serialized: value,
        }
    }

    fn clear(&mut self) {}
}

#[derive(Clone, Default)]
struct DebugMetricAccumulator {
    current: f64,
    sum: f64,
    count: usize,
}

impl DebugMetricAccumulator {
    fn update(&mut self, value: f64, unit: Option<&str>) -> SerializedEntry {
        self.current = value;
        if value.is_finite() {
            self.sum += value;
            self.count += 1;
        }
        SerializedEntry {
            formatted: format_metric_value(value, unit),
            serialized: value.to_string(),
        }
    }

    fn clear(&mut self) {
        *self = Self::default();
    }

    fn value(&self) -> NumericEntry {
        NumericEntry::Value(self.current)
    }

    fn running_value(&self) -> NumericEntry {
        if self.count == 0 {
            NumericEntry::Value(f64::NAN)
        } else {
            NumericEntry::Aggregated {
                aggregated_value: self.sum / self.count as f64,
                count: self.count,
            }
        }
    }
}

pub(super) fn register_burn_metrics(
    mut training: SupervisedTraining<CudaLearningComponents>,
) -> SupervisedTraining<CudaLearningComponents> {
    if !diagnostics::enabled() {
        return training;
    }

    for field in DEBUG_TEXT_FIELDS {
        training = training.metric_train(CudaDebugTextMetric::new(field.spec()));
    }
    for field in DEBUG_METRIC_FIELDS {
        training = training.metric_train_numeric(CudaDebugMetric::new(field.spec()));
    }
    training
}

pub(super) struct DebugTraceLogger {
    summary: Option<BufWriter<File>>,
    tensors: Option<BufWriter<File>>,
}

impl DebugTraceLogger {
    pub(super) fn new(directory: PathBuf) -> AppResult<Self> {
        if !diagnostics::enabled() {
            return Ok(Self {
                summary: None,
                tensors: None,
            });
        }

        fs::create_dir_all(&directory)?;
        let mut summary = BufWriter::new(File::create(directory.join("optimizer_summary.tsv"))?);
        let mut tensors = BufWriter::new(File::create(directory.join("optimizer_tensors.tsv"))?);
        writeln!(
            summary,
            "step\tsource\twindow_offset\tbatch_size\tseq_len\tupdate_count\tpositive_update_dot_count\tzero_grad_changed_count\tmax_update_to_weight_rms\tdlogits_rms\tdlogits_max\td_lm_head_rms\td_lm_head_max\td_embedding_rms\td_embedding_max\ttoken_embedding_global_before\ttoken_embedding_global_after\ttoken_embedding_changed_bytes\ttoken_embedding_hash_before\ttoken_embedding_hash_after"
        )?;
        writeln!(
            tensors,
            "step\tsource\twindow_offset\tbatch_size\tseq_len\ttensor\tlen\tgrad_rms\tgrad_max\tgrad_nonzero\tgrad_finite\tweight_rms_before\tweight_rms_after\tdelta_rms\tdelta_max\tupdate_to_weight_rms\tdelta_grad_dot\tdelta_grad_cos\tpredicted_delta_rms\tpredicted_delta_grad_dot\tpredicted_delta_grad_cos\tquant_error_rms\tquant_error_to_predicted_delta_rms\tchanged_bytes\tchanged_scales\tglobal_before\tglobal_after"
        )?;
        println!("debug_metrics_dir={}", directory.display());
        Ok(Self {
            summary: Some(summary),
            tensors: Some(tensors),
        })
    }

    pub(super) fn log_train_step(&mut self, step: usize, item: &CudaTrainOutput) -> AppResult {
        let Some(trace) = item.stats.diagnostics.as_ref() else {
            return Ok(());
        };

        if let Some(summary) = self.summary.as_mut() {
            writeln!(
                summary,
                "{step}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:016x}\t{:016x}",
                tsv_field(&item.source),
                item.window_offset,
                item.batch_size,
                item.seq_len,
                trace.update_count,
                trace.positive_update_dot_count,
                trace.zero_grad_changed_count,
                trace.max_update_to_weight_rms,
                trace.dlogits_rms,
                trace.dlogits_max,
                trace.d_lm_head_rms,
                trace.d_lm_head_max,
                trace.d_embedding_rms,
                trace.d_embedding_max,
                trace.token_embedding_global_before,
                trace.token_embedding_global_after,
                trace.token_embedding_changed_bytes,
                trace.token_embedding_hash_before,
                trace.token_embedding_hash_after,
            )?;
        }

        if let Some(tensors) = self.tensors.as_mut() {
            for update in &trace.updates {
                write_tensor_row(tensors, step, item, update)?;
            }
        }

        Ok(())
    }
}

fn write_tensor_row(
    writer: &mut BufWriter<File>,
    step: usize,
    item: &CudaTrainOutput,
    update: &TensorUpdateDiagnostics,
) -> AppResult {
    writeln!(
        writer,
        "{step}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        tsv_field(&item.source),
        item.window_offset,
        item.batch_size,
        item.seq_len,
        tsv_field(&update.name),
        update.len,
        update.grad_rms,
        update.grad_max,
        update.grad_nonzero,
        update.grad_finite,
        update.weight_rms_before,
        update.weight_rms_after,
        update.delta_rms,
        update.delta_max,
        update.update_to_weight_rms,
        update.delta_grad_dot,
        update.delta_grad_cos,
        update.predicted_delta_rms,
        update.predicted_delta_grad_dot,
        update.predicted_delta_grad_cos,
        update.quant_error_rms,
        update.quant_error_to_predicted_delta_rms,
        update.changed_bytes,
        update.changed_scales,
        update.global_before,
        update.global_after,
    )?;
    Ok(())
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

fn format_metric_value(value: f64, unit: Option<&str>) -> String {
    match unit {
        Some(unit) => format!("{value:.6} {unit}"),
        None => format!("{value:.6}"),
    }
}

fn tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}
