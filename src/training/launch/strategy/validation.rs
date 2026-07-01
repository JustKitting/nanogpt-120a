use std::time::Instant;

use burn::data::dataloader::Progress;
use burn::train::{
    EventProcessorTraining, LearnerEvent, SupervisedTrainingEventProcessor, TrainingItem,
    ValidLoader,
};

use super::super::super::Trainer;
use super::super::{
    data_loader::CudaValidationInput, metrics::CudaValidOutput, CudaLearningComponents,
};
use super::epoch_progress;
use crate::AppResult;

pub(super) fn process_validation(
    trainer: &mut Trainer,
    processor: &mut SupervisedTrainingEventProcessor<CudaLearningComponents>,
    validation: &CudaValidationInput,
    step: usize,
    completed_steps: usize,
) -> AppResult<CudaValidOutput> {
    let eval_start = Instant::now();
    let val_loss = trainer.eval_loss_windows(&validation.tokens, validation.window_count)?;
    let output = CudaValidOutput {
        val_loss,
        eval_elapsed_s: eval_start.elapsed().as_secs_f64(),
        window_count: validation.window_count,
        completed_steps,
    };
    processor.process_valid(LearnerEvent::ProcessedItem(TrainingItem::new(
        output.clone(),
        Progress::new(validation.window_count, validation.window_count),
        epoch_progress(),
        Some(step),
        None,
    )));
    Ok(output)
}

pub(super) fn validation_input(
    dataloader_valid: ValidLoader<CudaLearningComponents>,
) -> AppResult<CudaValidationInput> {
    match dataloader_valid.iter().next() {
        Some(Ok(input)) => Ok(input),
        Some(Err(err)) => Err(format!("validation dataloader failed: {err}").into()),
        None => Err("validation dataloader produced no windows".into()),
    }
}
