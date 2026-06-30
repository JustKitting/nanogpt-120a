use std::sync::Arc;

use burn::train::SupervisedTraining;
use burn::train::metric::{
    Adaptor, Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes,
    NumericEntry, SerializedEntry,
};

use super::super::TrainStats;
use super::CudaLearningComponents;

#[derive(Clone)]
pub(in crate::training) struct CudaTrainOutput {
    pub(in crate::training) source: String,
    pub(in crate::training) window_offset: usize,
    pub(in crate::training) batch_size: usize,
    pub(in crate::training) seq_len: usize,
    pub(in crate::training) stats: Arc<TrainStats>,
}

impl burn::train::ItemLazy for CudaTrainOutput {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl Adaptor<CudaTrainOutput> for CudaTrainOutput {
    fn adapt(&self) -> CudaTrainOutput {
        self.clone()
    }
}

#[derive(Clone)]
pub(in crate::training) struct CudaValidOutput {
    pub(super) val_loss: f32,
    pub(super) eval_elapsed_s: f64,
    pub(super) window_count: usize,
    pub(super) completed_steps: usize,
}

impl burn::train::ItemLazy for CudaValidOutput {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl Adaptor<CudaValidOutput> for CudaValidOutput {
    fn adapt(&self) -> CudaValidOutput {
        self.clone()
    }
}

#[derive(Clone, Copy)]
struct TrainMetricSpec {
    name: &'static str,
    unit: Option<&'static str>,
    higher_is_better: bool,
    field: TrainMetricField,
}

metric_fields! {
    TrainMetricField, TRAIN_METRIC_FIELDS, TrainMetricSpec {
        Loss => ("Loss", None, false),
        ForwardMs => ("Forward", Some("ms"), false),
        BackwardEnqueueMs => ("Backward enqueue", Some("ms"), false),
        LossHostWaitMs => ("Loss host wait", Some("ms"), false),
        OptimizerMs => ("Optimizer", Some("ms"), false),
        AuroraMs => ("Aurora", Some("ms"), false),
        KdaClipMs => ("KDA clip", Some("ms"), false),
        AdamMs => ("Adam", Some("ms"), false),
        EmbeddingLookupMs => ("Embedding lookup", Some("ms"), false),
        TokenEmbeddingMs => ("Token embedding", Some("ms"), false),
        FinalNormMs => ("Final norm", Some("ms"), false),
        BlocksMs => ("Blocks", Some("ms"), false),
        GradNorm => ("Grad norm", None, false),
        AdamLr => ("Adam LR", None, false),
        AuroraLr => ("Aurora LR", None, false),
        Tokens => ("Tokens", None, true),
        Logits => ("Logits", None, true),
        Finite => ("Finite", None, true),
        Nonzero => ("Nonzero", None, true),
        UpdateSkipped => ("Update skipped", None, false),
        SkipLossSpike => ("Skip loss spike", None, false),
        SkipGradNormSpike => ("Skip grad norm spike", None, false),
        SkipNonFinite => ("Skip non finite", None, false),
        WindowOffset => ("Window offset", None, true),
        BatchSize => ("Batch size", None, true),
        SeqLen => ("Seq len", None, true),
    }
}

impl TrainMetricField {
    fn value(self, item: &CudaTrainOutput) -> f64 {
        let stats = item.stats.as_ref();
        let bool_value = |value: bool| if value { 1.0 } else { 0.0 };
        match self {
            Self::Loss => stats.loss as f64,
            Self::ForwardMs => stats.forward_ms,
            Self::BackwardEnqueueMs => stats.backward_enqueue_ms,
            Self::LossHostWaitMs => stats.loss_host_wait_ms,
            Self::OptimizerMs => stats.optimizer_ms,
            Self::AuroraMs => stats.optimizer.aurora_ms,
            Self::KdaClipMs => stats.optimizer.kda_clip_ms,
            Self::AdamMs => stats.optimizer.adam_ms,
            Self::EmbeddingLookupMs => stats.optimizer.embedding_lookup_ms,
            Self::TokenEmbeddingMs => stats.optimizer.token_embedding_ms,
            Self::FinalNormMs => stats.optimizer.final_norm_ms,
            Self::BlocksMs => stats.optimizer.blocks_ms,
            Self::GradNorm => stats.optimizer.grad_norm as f64,
            Self::AdamLr => stats.optimizer.adam_lr as f64,
            Self::AuroraLr => stats.optimizer.aurora_lr as f64,
            Self::Tokens => stats.tokens as f64,
            Self::Logits => stats.logits as f64,
            Self::Finite => bool_value(stats.finite),
            Self::Nonzero => bool_value(stats.nonzero),
            Self::UpdateSkipped => bool_value(stats.optimizer.update_skipped),
            Self::SkipLossSpike => bool_value(stats.optimizer.skip_loss_spike),
            Self::SkipGradNormSpike => bool_value(stats.optimizer.skip_grad_norm_spike),
            Self::SkipNonFinite => bool_value(stats.optimizer.skip_non_finite),
            Self::WindowOffset => item.window_offset as f64,
            Self::BatchSize => item.batch_size as f64,
            Self::SeqLen => item.seq_len as f64,
        }
    }
}

#[derive(Clone)]
struct CudaTrainMetric {
    spec: TrainMetricSpec,
    state: MetricAccumulator,
}

impl CudaTrainMetric {
    fn new(spec: TrainMetricSpec) -> Self {
        Self {
            spec,
            state: MetricAccumulator::default(),
        }
    }
}

impl Metric for CudaTrainMetric {
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

impl Numeric for CudaTrainMetric {
    fn value(&self) -> NumericEntry {
        self.state.value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

#[derive(Clone)]
struct CudaSourceMetric {
    name: Arc<String>,
}

impl CudaSourceMetric {
    fn new() -> Self {
        Self {
            name: Arc::new("Source".to_string()),
        }
    }
}

impl Metric for CudaSourceMetric {
    type Input = CudaTrainOutput;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        SerializedEntry {
            formatted: item.source.clone(),
            serialized: item.source.clone(),
        }
    }

    fn clear(&mut self) {}
}

#[derive(Clone, Copy)]
struct ValidMetricSpec {
    name: &'static str,
    unit: Option<&'static str>,
    higher_is_better: bool,
    field: ValidMetricField,
}

metric_fields! {
    ValidMetricField, VALID_METRIC_FIELDS, ValidMetricSpec {
        Loss => ("Validation loss", None, false),
        EvalElapsed => ("Eval elapsed", Some("s"), false),
        WindowCount => ("Val windows", None, true),
        CompletedSteps => ("Completed steps", None, true),
    }
}

impl ValidMetricField {
    fn value(self, item: &CudaValidOutput) -> f64 {
        match self {
            Self::Loss => item.val_loss as f64,
            Self::EvalElapsed => item.eval_elapsed_s,
            Self::WindowCount => item.window_count as f64,
            Self::CompletedSteps => item.completed_steps as f64,
        }
    }
}

#[derive(Clone)]
struct CudaValidMetric {
    spec: ValidMetricSpec,
    state: MetricAccumulator,
}

impl CudaValidMetric {
    fn new(spec: ValidMetricSpec) -> Self {
        Self {
            spec,
            state: MetricAccumulator::default(),
        }
    }
}

impl Metric for CudaValidMetric {
    type Input = CudaValidOutput;

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

impl Numeric for CudaValidMetric {
    fn value(&self) -> NumericEntry {
        self.state.value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

#[derive(Clone, Default)]
struct MetricAccumulator {
    current: f64,
    sum: f64,
    count: usize,
}

impl MetricAccumulator {
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

fn format_metric_value(value: f64, unit: Option<&str>) -> String {
    match unit {
        Some(unit) => format!("{value:.6} {unit}"),
        None => format!("{value:.6}"),
    }
}

pub(super) fn register_cuda_metrics(
    mut training: SupervisedTraining<CudaLearningComponents>,
) -> SupervisedTraining<CudaLearningComponents> {
    training = training.metric_train(CudaSourceMetric::new());
    for field in TRAIN_METRIC_FIELDS {
        training = training.metric_train_numeric(CudaTrainMetric::new(field.spec()));
    }
    for field in VALID_METRIC_FIELDS {
        training = training.metric_valid_numeric(CudaValidMetric::new(field.spec()));
    }
    training
}
