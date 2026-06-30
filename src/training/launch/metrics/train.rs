use std::sync::Arc;

use burn::train::SupervisedTraining;
use burn::train::metric::{
    Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry,
    SerializedEntry,
};

use super::output::CudaTrainOutput;
use crate::training::launch::CudaLearningComponents;
use crate::training::metric_accumulator::MetricAccumulator;

mod fields;

use fields::{TrainMetricSpec, train_metric_specs};

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

pub(super) fn register_train_metrics(
    mut training: SupervisedTraining<CudaLearningComponents>,
) -> SupervisedTraining<CudaLearningComponents> {
    training = training.metric_train(CudaSourceMetric::new());
    for spec in train_metric_specs() {
        training = training.metric_train_numeric(CudaTrainMetric::new(spec));
    }
    training
}
