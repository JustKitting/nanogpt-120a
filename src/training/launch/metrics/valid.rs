use burn::train::SupervisedTraining;

use crate::training::launch::CudaLearningComponents;
use crate::training::numeric_metric::CudaNumericMetric;

mod fields;

use fields::valid_metric_specs;

pub(super) fn register_valid_metrics(
    mut training: SupervisedTraining<CudaLearningComponents>,
) -> SupervisedTraining<CudaLearningComponents> {
    for spec in valid_metric_specs() {
        training = training.metric_valid_numeric(CudaNumericMetric::new(spec));
    }
    training
}
