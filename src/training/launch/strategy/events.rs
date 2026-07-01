use std::sync::Arc;

use burn::data::dataloader::Progress;
use burn::train::{
    EventProcessorTraining, LearnerEvent, SupervisedTrainingEventProcessor, TrainingItem,
};

use super::{epoch_progress, CudaLearningComponents};
use crate::training::data::TokenWindowBatch;
use crate::training::debug_metrics::DebugTraceLogger;
use crate::training::launch::metrics::{CudaTrainOutput, CudaValidOutput};
use crate::training::TrainStats;
use crate::AppResult;

pub(super) fn process_train_step(
    logger: &mut DebugTraceLogger,
    processor: &mut SupervisedTrainingEventProcessor<CudaLearningComponents>,
    step: usize,
    window: &TokenWindowBatch,
    stats: TrainStats,
    progress: Progress,
) -> AppResult {
    let output = CudaTrainOutput {
        source: window.source.display().to_string(),
        window_offset: window.offset,
        batch_size: window.batch_size,
        seq_len: window.seq_len,
        stats: Arc::new(stats),
    };
    logger.log_train_step(step, &output)?;
    processor.process_train(LearnerEvent::ProcessedItem(TrainingItem::new(
        output,
        progress,
        epoch_progress(),
        Some(step),
        None,
    )));
    Ok(())
}

pub(super) fn process_valid_step(
    processor: &mut SupervisedTrainingEventProcessor<CudaLearningComponents>,
    step: usize,
    output: CudaValidOutput,
) -> CudaValidOutput {
    processor.process_valid(LearnerEvent::ProcessedItem(TrainingItem::new(
        output.clone(),
        Progress::new(output.window_count, output.window_count),
        epoch_progress(),
        Some(step),
        None,
    )));
    output
}
