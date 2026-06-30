use burn::train::SupervisedTraining;

use super::diagnostics;
use super::launch::CudaLearningComponents;

mod numeric;
mod text;
mod trace;

pub(super) use trace::DebugTraceLogger;

pub(super) fn register_burn_metrics(
    mut training: SupervisedTraining<CudaLearningComponents>,
) -> SupervisedTraining<CudaLearningComponents> {
    if !diagnostics::enabled() {
        return training;
    }

    training = text::register_text_metrics(training);
    numeric::register_numeric_metrics(training)
}
