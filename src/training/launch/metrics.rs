use burn::train::SupervisedTraining;

use super::CudaLearningComponents;

mod accumulator;
mod output;
mod train;
mod valid;

pub(in crate::training) use output::{CudaTrainOutput, CudaValidOutput};

pub(super) fn register_cuda_metrics(
    mut training: SupervisedTraining<CudaLearningComponents>,
) -> SupervisedTraining<CudaLearningComponents> {
    training = train::register_train_metrics(training);
    valid::register_valid_metrics(training)
}
