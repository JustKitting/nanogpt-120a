use std::sync::Arc;

use burn::train::{
    EventProcessorTraining, Interrupter, LearnerEvent, SupervisedTrainingEventProcessor,
    TrainLoader, TrainingItem, ValidLoader,
};

use super::{
    CudaTrainingStrategy,
    artifacts::finish_training_artifacts,
    budget::WallClockBudget,
    progress::{TRAIN_EPOCH, epoch_progress},
    validation::{process_validation, validation_input},
};
use crate::AppResult;
use crate::training::launch::{
    CudaLearningComponents,
    config::{load_model_path, should_eval_step, should_log_step},
    metrics::CudaTrainOutput,
};
use crate::training::{Trainer, debug_metrics};

impl CudaTrainingStrategy {
    pub(super) fn run_training(
        &self,
        dataloader_train: TrainLoader<CudaLearningComponents>,
        dataloader_valid: ValidLoader<CudaLearningComponents>,
        processor: &mut SupervisedTrainingEventProcessor<CudaLearningComponents>,
        interrupter: &Interrupter,
    ) -> AppResult {
        let mut trainer = Trainer::new(self.config.seed)?;
        if let Some(path) = load_model_path() {
            trainer.load_model(&path)?;
            println!("loaded_model={}", path.display());
        }

        let validation = validation_input(dataloader_valid)?;
        let mut train_batch = trainer.reusable_default_batch()?;
        let mut debug_logger = debug_metrics::DebugTraceLogger::new(self.run_output.path("debug"))?;
        let wall_clock = WallClockBudget::new(self.config.max_seconds);
        let mut completed_steps = 0usize;
        let mut train_iter = dataloader_train.iter();

        while let Some(item) = train_iter.next() {
            let step = completed_steps;
            let log_step = should_log_step(step, self.config.step_cap, self.config.log_interval);
            let window = item.map_err(|err| format!("training dataloader failed: {err}"))?;
            let source = window.source.display().to_string();
            let batch = trainer.upload_default_batch(&mut train_batch, &window.tokens)?;
            let stats = trainer.train_step(batch, log_step)?;
            completed_steps = step + 1;

            if log_step {
                let output = CudaTrainOutput {
                    source,
                    window_offset: window.offset,
                    batch_size: window.batch_size,
                    seq_len: window.seq_len,
                    stats: Arc::new(stats),
                };
                debug_logger.log_train_step(step, &output)?;
                processor.process_train(LearnerEvent::ProcessedItem(TrainingItem::new(
                    output,
                    train_iter.progress(),
                    epoch_progress(),
                    Some(step),
                    None,
                )));
            }

            if should_eval_step(step, self.config.step_cap, self.config.eval_interval) {
                process_validation(&mut trainer, processor, &validation, step, completed_steps)?;
            }

            let stopped_by_burn = interrupter.should_stop();
            if stopped_by_burn || wall_clock.expired() {
                let reason = if stopped_by_burn { "burn_interrupter" } else { "wall_clock" };
                println!(
                    "stopped_by_{reason}=true elapsed_s={:.3} completed_steps={completed_steps}",
                    wall_clock.elapsed_seconds(),
                );
                break;
            }
        }
        let train_elapsed_s = wall_clock.elapsed_seconds();

        let final_eval = process_validation(
            &mut trainer,
            processor,
            &validation,
            completed_steps.saturating_sub(1),
            completed_steps,
        )?;
        processor.process_train(LearnerEvent::EndEpoch(TRAIN_EPOCH));
        processor.process_valid(LearnerEvent::EndEpoch(TRAIN_EPOCH));
        println!(
            "heldout_eval split=val val_loss={:.6} train_elapsed_s={:.3} eval_elapsed_s={:.3} completed_steps={completed_steps}",
            final_eval.val_loss, train_elapsed_s, final_eval.eval_elapsed_s,
        );

        finish_training_artifacts(
            &mut trainer,
            &self.dataset,
            train_elapsed_s,
            &self.run_output,
        )
    }
}
