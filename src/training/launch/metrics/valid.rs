use std::sync::Arc;

use burn::train::SupervisedTraining;
use burn::train::metric::{
    Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry,
    SerializedEntry,
};

use super::output::CudaValidOutput;
use crate::training::launch::CudaLearningComponents;
use crate::training::metric_accumulator::MetricAccumulator;

mod fields;

use fields::{ValidMetricSpec, valid_metric_specs};

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
    for spec in valid_metric_specs() {
        training = training.metric_valid_numeric(CudaValidMetric::new(spec));
    }
    training
}
