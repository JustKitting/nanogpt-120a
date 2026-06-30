use std::sync::{Arc, Mutex};

use burn::train::{
    Learner, SupervisedLearningStrategy, SupervisedTrainingEventProcessor, TrainLoader,
    TrainingComponents, TrainingModel, ValidLoader,
};

use super::output::RunOutput;
use super::{CudaLearningComponents, TrainConfig};

mod artifacts;
mod budget;
mod progress;
mod run;
mod validation;

pub(super) struct CudaTrainingStrategy {
    dataset: String,
    config: TrainConfig,
    run_output: RunOutput,
    result: Arc<Mutex<Option<Result<(), String>>>>,
}

impl CudaTrainingStrategy {
    pub(super) fn new(
        dataset: String,
        config: TrainConfig,
        run_output: RunOutput,
        result: Arc<Mutex<Option<Result<(), String>>>>,
    ) -> Self {
        Self {
            dataset,
            config,
            run_output,
            result,
        }
    }
}

impl SupervisedLearningStrategy<CudaLearningComponents> for CudaTrainingStrategy {
    fn fit(
        &self,
        training_components: TrainingComponents<CudaLearningComponents>,
        learner: Learner<CudaLearningComponents>,
        dataloader_train: TrainLoader<CudaLearningComponents>,
        dataloader_valid: ValidLoader<CudaLearningComponents>,
        _starting_epoch: usize,
    ) -> (
        TrainingModel<CudaLearningComponents>,
        SupervisedTrainingEventProcessor<CudaLearningComponents>,
    ) {
        let mut processor = training_components.event_processor;
        let result = self.run_training(
            dataloader_train,
            dataloader_valid,
            &mut processor,
            &training_components.interrupter,
        );
        *self.result.lock().unwrap() = Some(result.map_err(|err| err.to_string()));
        (learner.model(), processor)
    }
}
