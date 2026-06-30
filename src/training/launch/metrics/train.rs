use burn::train::SupervisedTraining;

use super::output::CudaTrainOutput;
use crate::training::launch::CudaLearningComponents;
use crate::training::numeric_metric::CudaNumericMetric;
use crate::training::text_metric::{CudaTextMetric, TextMetricSpec};

mod fields;

use fields::train_metric_specs;

#[derive(Clone, Copy)]
struct SourceMetricSpec;

impl TextMetricSpec for SourceMetricSpec {
    type Input = CudaTrainOutput;

    fn name(self) -> &'static str {
        "Source"
    }

    fn value(self, item: &CudaTrainOutput) -> String {
        item.source.clone()
    }
}

pub(super) fn register_train_metrics(
    mut training: SupervisedTraining<CudaLearningComponents>,
) -> SupervisedTraining<CudaLearningComponents> {
    training = training.metric_train(CudaTextMetric::new(SourceMetricSpec));
    for spec in train_metric_specs() {
        training = training.metric_train_numeric(CudaNumericMetric::new(spec));
    }
    training
}
