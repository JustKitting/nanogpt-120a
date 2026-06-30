use std::sync::{Arc, Mutex};

use super::{TokenDataLoader, debug_metrics};
use crate::AppResult;
use burn::data::dataloader::DataLoader;
use burn::train::logger::FileMetricLogger;
use burn::train::{Interrupter, Learner, SupervisedTraining, TrainingStrategy};

mod burn_shim;
mod config;
mod data_loader;
mod metrics;
mod output;
mod render;
mod strategy;

pub(super) use burn_shim::CudaLearningComponents;
use burn_shim::{
    BurnBackend, BurnInnerBackend, CudaBurnModel, CudaNoopOptimizer, CudaTrainInput, CudaValidInput,
};
pub(super) use config::{TrainConfig, env_bool, env_nonempty};
use data_loader::{CudaTrainDataLoader, CudaValidDataLoader};
pub(super) use metrics::CudaTrainOutput;
use metrics::register_cuda_metrics;
use output::{RunOutput, build_run_info};
use render::{BoxedMetricsRenderer, default_renderer};
use strategy::CudaTrainingStrategy;

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
        let strategy = Arc::new(CudaTrainingStrategy::new(
            self.dataset.clone(),
            self.config,
            run_output.clone(),
            Arc::clone(&strategy_result),
        ));

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
