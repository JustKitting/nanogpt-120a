use std::sync::Arc;

use burn::train::SupervisedTraining;
use burn::train::metric::{
    Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry,
    SerializedEntry,
};

use super::super::launch::{CudaLearningComponents, CudaTrainOutput};
use crate::training::metric_accumulator::MetricAccumulator;

mod fields;

use fields::{DebugMetricSpec, debug_metric_specs};

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
        Arc::new(self.spec.name().to_string())
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: self.spec.unit().map(str::to_string),
            higher_is_better: self.spec.higher_is_better(),
        }
        .into()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        self.state.update(self.spec.value(item), self.spec.unit())
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
    for spec in debug_metric_specs() {
        training = training.metric_train_numeric(CudaDebugMetric::new(spec));
    }
    training
}
