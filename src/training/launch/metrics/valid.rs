use std::sync::Arc;

use burn::train::SupervisedTraining;
use burn::train::metric::{
    Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry,
    SerializedEntry,
};

use super::accumulator::MetricAccumulator;
use super::output::CudaValidOutput;
use crate::training::launch::CudaLearningComponents;

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

pub(super) fn register_valid_metrics(
    mut training: SupervisedTraining<CudaLearningComponents>,
) -> SupervisedTraining<CudaLearningComponents> {
    for field in VALID_METRIC_FIELDS {
        training = training.metric_valid_numeric(CudaValidMetric::new(field.spec()));
    }
    training
}
