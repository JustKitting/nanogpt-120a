use burn::train::SupervisedTraining;

use super::super::launch::CudaLearningComponents;
use crate::training::numeric_metric::CudaNumericMetric;

mod fields;

use fields::debug_metric_specs;

pub(super) fn register_numeric_metrics(
    mut training: SupervisedTraining<CudaLearningComponents>,
) -> SupervisedTraining<CudaLearningComponents> {
    for spec in debug_metric_specs() {
        training = training.metric_train_numeric(CudaNumericMetric::new(spec));
    }
    training
}
