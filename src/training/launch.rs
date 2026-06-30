use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::{TokenDataLoader, Trainer, debug_metrics};
use crate::AppResult;
use burn::data::dataloader::{DataLoader, Progress};
use burn::train::logger::FileMetricLogger;
use burn::train::{
    EventProcessorTraining, Interrupter, Learner, LearnerEvent, SupervisedLearningStrategy,
    SupervisedTraining, SupervisedTrainingEventProcessor, TrainLoader, TrainingComponents,
    TrainingItem, TrainingModel, TrainingStrategy, ValidLoader,
};

mod burn_shim;
mod config;
mod data_loader;
mod metrics;
mod output;
mod render;

pub(super) use burn_shim::CudaLearningComponents;
use burn_shim::{
    BurnBackend, BurnInnerBackend, CudaBurnModel, CudaNoopOptimizer, CudaTrainInput, CudaValidInput,
};
pub(super) use config::{TrainConfig, env_bool, env_nonempty};
use config::{
    generate_prompt, generate_tokens, load_model_path, sampling_config, should_eval_step,
    should_log_step,
};
use data_loader::{CudaTrainDataLoader, CudaValidDataLoader, CudaValidationInput};
use metrics::register_cuda_metrics;
pub(super) use metrics::{CudaTrainOutput, CudaValidOutput};
use output::{RunOutput, build_run_info, ensure_parent, save_model_path, write_generated_text};
use render::{BoxedMetricsRenderer, default_renderer};

const TRAIN_EPOCH: usize = 1;

pub(crate) fn launch_from_env() -> AppResult {
    let dataset = TokenDataLoader::training_dataset_name();
    let config = TrainConfig::from_env();
    BurnTrainingLauncher::new(dataset, config).run()
}

struct BurnTrainingLauncher {
    dataset: String,
    config: TrainConfig,
    interrupter: Interrupter,
}

impl BurnTrainingLauncher {
    fn new(dataset: String, config: TrainConfig) -> Self {
        let interrupter = Interrupter::new();
        Self {
            dataset,
            config,
            interrupter,
        }
    }

    fn run(self) -> AppResult {
        let run_label = format!("{}s", self.config.max_seconds.round() as u64);
        let run_output = RunOutput::new(&self.dataset, &run_label)?;
        let metrics_dir = run_output.path("metrics");
        println!("run_dir={}", run_output.dir().display());
        println!("metrics_dir={}", metrics_dir.display());

        let data = TokenDataLoader::from_training_dataset()?;
        let training_tokens = data.token_count();
        let validation_tokens = data.validation_tokens()?;
        let validation_window_count = TokenDataLoader::validation_window_count();

        run_output.write_info(&build_run_info(&self.dataset, &self.config))?;
        println!(
            "training_tokens={} max_seconds={:.3} step_cap={}",
            training_tokens, self.config.max_seconds, self.config.step_cap
        );

        let train_loader: Arc<dyn DataLoader<BurnBackend, CudaTrainInput>> =
            Arc::new(CudaTrainDataLoader::new(data, self.config.step_cap));
        let valid_loader: Arc<dyn DataLoader<BurnInnerBackend, CudaValidInput>> = Arc::new(
            CudaValidDataLoader::new(validation_tokens, validation_window_count),
        );
        let strategy_result = Arc::new(Mutex::new(None));
        let strategy = Arc::new(CudaTrainingStrategy {
            dataset: self.dataset.clone(),
            config: self.config,
            run_output: run_output.clone(),
            result: Arc::clone(&strategy_result),
        });

        let training = SupervisedTraining::<CudaLearningComponents>::new(
            run_output.dir(),
            train_loader,
            valid_loader,
        )
        .with_interrupter(self.interrupter.clone())
        .with_metric_logger(FileMetricLogger::new(metrics_dir))
        .with_application_logger(None)
        .renderer(BoxedMetricsRenderer::new(default_renderer(
            self.interrupter.clone(),
        )))
        .with_training_strategy(TrainingStrategy::Custom(strategy));
        let training = debug_metrics::register_burn_metrics(register_cuda_metrics(training));
        let burn_device = Default::default();
        let learner = Learner::new(CudaBurnModel::new(&burn_device), CudaNoopOptimizer, 0.0);
        let _learning_result = training.launch(learner);

        match strategy_result.lock().unwrap().take() {
            Some(Ok(())) => Ok(()),
            Some(Err(err)) => Err(err.into()),
            None => Err("Burn custom training strategy did not report a result".into()),
        }
    }
}

struct CudaTrainingStrategy {
    dataset: String,
    config: TrainConfig,
    run_output: RunOutput,
    result: Arc<Mutex<Option<Result<(), String>>>>,
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

impl CudaTrainingStrategy {
    fn run_training(
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
            let stats = trainer.train_step(&batch, log_step)?;
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
                self.process_validation(
                    &mut trainer,
                    processor,
                    &validation,
                    step,
                    completed_steps,
                )?;
            }

            if interrupter.should_stop() {
                println!(
                    "stopped_by_burn_interrupter=true elapsed_s={:.3} completed_steps={completed_steps}",
                    wall_clock.elapsed_seconds(),
                );
                break;
            }
            if wall_clock.expired() {
                println!(
                    "stopped_by_wall_clock=true elapsed_s={:.3} completed_steps={completed_steps}",
                    wall_clock.elapsed_seconds(),
                );
                break;
            }
        }
        let train_elapsed_s = wall_clock.elapsed_seconds();

        let final_eval = self.process_validation(
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

        if let Some(path) = save_model_path(&self.run_output) {
            ensure_parent(&path)?;
            trainer.save_model(&path)?;
            println!("saved_model={}", path.display());
        }

        if let Some(prompt) = generate_prompt(&self.dataset, train_elapsed_s) {
            let text = trainer.generate_sampled(&prompt, generate_tokens(), sampling_config())?;
            let generated_path = write_generated_text(&self.run_output, &text)?;
            println!("generated_text={}", generated_path.display());
            println!("generated_text_begin");
            println!("{text}");
            println!("generated_text_end");
        }

        Ok(())
    }

    fn process_validation(
        &self,
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
}

fn validation_input(
    dataloader_valid: ValidLoader<CudaLearningComponents>,
) -> AppResult<CudaValidationInput> {
    match dataloader_valid.iter().next() {
        Some(Ok(input)) => Ok(input),
        Some(Err(err)) => Err(format!("validation dataloader failed: {err}").into()),
        None => Err("validation dataloader produced no windows".into()),
    }
}

fn epoch_progress() -> Progress {
    Progress::new(TRAIN_EPOCH, TRAIN_EPOCH)
}

struct WallClockBudget {
    start: Instant,
    max: Duration,
}

impl WallClockBudget {
    fn new(max_seconds: f64) -> Self {
        Self {
            start: Instant::now(),
            max: Duration::from_secs_f64(max_seconds),
        }
    }

    fn elapsed_seconds(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn expired(&self) -> bool {
        self.start.elapsed() >= self.max
    }
}
