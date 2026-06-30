use std::sync::Arc;

use burn::train::SupervisedTraining;
use burn::train::metric::{Metric, MetricMetadata, MetricName, SerializedEntry};

use super::output::CudaTrainOutput;
use crate::training::launch::CudaLearningComponents;
use crate::training::numeric_metric::CudaNumericMetric;

mod fields;

use fields::train_metric_specs;

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
        training = training.metric_train_numeric(CudaNumericMetric::new(spec));
    }
    training
}
