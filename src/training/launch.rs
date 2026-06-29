use std::collections::HashSet;
use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::module::{EmptyRecord, Module, Param};
use burn::optim::{GradientsParams, LearningRate, MultiGradientsParams, Optimizer};
use burn::tensor::Tensor;
use burn::tensor::backend::{Backend, BackendTypes};
use burn::train::logger::FileMetricLogger;
use burn::train::metric::{
    Adaptor, Metric, MetricAttributes, MetricDefinition, MetricId, MetricMetadata, MetricName,
    Numeric, NumericAttributes, NumericEntry, SerializedEntry,
};
use burn::train::renderer::tui::TuiMetricsRendererWrapper;
use burn::train::renderer::{
    CliMetricsRenderer, EvaluationName, EvaluationProgress, MetricState, MetricsRenderer,
    ProgressType, TrainingProgress,
};
use burn::train::{
    EventProcessorTraining, InferenceStep, Interrupter, Learner, LearnerEvent,
    LearningComponentsMarker, SupervisedLearningStrategy, SupervisedTraining,
    SupervisedTrainingEventProcessor, TrainLoader, TrainOutput, TrainStep, TrainingComponents,
    TrainingItem, TrainingModel, TrainingStrategy, ValidLoader,
};
use gpt2_nvfp4::{
    GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_N_LAYER, GPT2_SEQ_LEN, GPT2_TOKEN_ROWS,
};
use rust_kernels_cuda::optimizer::{AURORA_COOPERATIVE_BLOCKS, AURORA_MATRIX_PHASES};
use time::OffsetDateTime;

use super::{SamplingConfig, TokenDataLoader, TrainStats, Trainer, debug_metrics};
use crate::AppResult;

const DEFAULT_SEED: u64 = 0x4750_5432;
const DEFAULT_TRAIN_MAX_SECONDS: f64 = 900.0;
const DEFAULT_TRAIN_STEP_CAP: usize = 1_000_000;
const AUTO_GENERATE_MIN_TRAIN_SECONDS: f64 = 900.0;
const DEFAULT_SYNTH_PROMPT: &str = "The";
const DEFAULT_SHAKESPEARE_PROMPT: &str = "KING:";
const RUNS_DIR: &str = "target/runs";
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

#[derive(Clone, Copy)]
struct TrainConfig {
    seed: u64,
    step_cap: usize,
    log_interval: usize,
    eval_interval: Option<usize>,
    max_seconds: f64,
}

impl TrainConfig {
    fn from_env() -> Self {
        Self {
            seed: env_u64("TRAIN_SEED").unwrap_or(DEFAULT_SEED),
            step_cap: env_usize("TRAIN_STEPS").unwrap_or(DEFAULT_TRAIN_STEP_CAP),
            log_interval: env_usize("TRAIN_LOG_INTERVAL").unwrap_or(1).max(1),
            eval_interval: env_usize("TRAIN_EVAL_INTERVAL").filter(|interval| *interval > 0),
            max_seconds: env_f64("TRAIN_MAX_SECONDS")
                .filter(|seconds| *seconds > 0.0)
                .unwrap_or(DEFAULT_TRAIN_MAX_SECONDS),
        }
    }
}

type BurnInnerBackend = burn::backend::NdArray;
type BurnBackend = burn::backend::Autodiff<BurnInnerBackend>;
type CudaBurnModel = CudaBurnModule<BurnBackend>;
pub(super) type CudaLearningComponents =
    LearningComponentsMarker<BurnBackend, LearningRate, CudaBurnModel, CudaNoopOptimizer>;
type CudaTrainInput = Result<super::data::TokenWindowBatch, String>;
type CudaValidInput = Result<CudaValidationInput, String>;

#[derive(Module, Debug)]
pub(super) struct CudaBurnModule<B: Backend> {
    marker: Param<Tensor<B, 1>>,
}

impl<B: Backend> CudaBurnModule<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            marker: Param::from_data([0.0_f32], device),
        }
    }
}

impl Default for CudaBurnModule<BurnBackend> {
    fn default() -> Self {
        Self::new(&Default::default())
    }
}

impl TrainStep for CudaBurnModule<BurnBackend> {
    type Input = CudaTrainInput;
    type Output = CudaTrainOutput;

    fn step(&self, _item: Self::Input) -> TrainOutput<Self::Output> {
        panic!("CudaBurnModel::step must not be called; CudaTrainingStrategy owns training")
    }
}

impl InferenceStep for CudaBurnModule<BurnInnerBackend> {
    type Input = CudaValidInput;
    type Output = CudaValidOutput;

    fn step(&self, _item: Self::Input) -> Self::Output {
        panic!("CudaBurnModel::step must not be called; CudaTrainingStrategy owns validation")
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct CudaNoopOptimizer;

impl Optimizer<CudaBurnModel, BurnBackend> for CudaNoopOptimizer {
    type Record = EmptyRecord;

    fn step(
        &mut self,
        _lr: LearningRate,
        module: CudaBurnModel,
        _grads: GradientsParams,
    ) -> CudaBurnModel {
        module
    }

    fn step_multi(
        &mut self,
        _lr: LearningRate,
        module: CudaBurnModel,
        _grads: MultiGradientsParams,
    ) -> CudaBurnModel {
        module
    }

    fn to_record(&self) -> Self::Record {
        EmptyRecord::new()
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }
}

#[derive(Clone)]
struct CudaTrainDataLoader {
    data: Arc<Mutex<TokenDataLoader>>,
    total_steps: usize,
}

impl CudaTrainDataLoader {
    fn new(data: TokenDataLoader, total_steps: usize) -> Self {
        Self {
            data: Arc::new(Mutex::new(data)),
            total_steps,
        }
    }
}

impl DataLoader<BurnBackend, CudaTrainInput> for CudaTrainDataLoader {
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<CudaTrainInput> + 'a> {
        Box::new(CudaTrainIterator {
            data: Arc::clone(&self.data),
            produced: 0,
            total_steps: self.total_steps,
        })
    }

    fn num_items(&self) -> usize {
        self.total_steps
    }

    fn to_device(
        &self,
        _device: &<BurnBackend as BackendTypes>::Device,
    ) -> Arc<dyn DataLoader<BurnBackend, CudaTrainInput>> {
        Arc::new(self.clone())
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<BurnBackend, CudaTrainInput>> {
        let mut sliced = self.clone();
        sliced.total_steps = end.saturating_sub(start).min(self.total_steps);
        Arc::new(sliced)
    }
}

struct CudaTrainIterator {
    data: Arc<Mutex<TokenDataLoader>>,
    produced: usize,
    total_steps: usize,
}

impl Iterator for CudaTrainIterator {
    type Item = CudaTrainInput;

    fn next(&mut self) -> Option<Self::Item> {
        if self.produced >= self.total_steps {
            return None;
        }
        self.produced += 1;
        Some(
            self.data
                .lock()
                .map_err(|err| err.to_string())
                .and_then(|mut data| data.next_batch().map_err(|err| err.to_string())),
        )
    }
}

impl DataLoaderIterator<CudaTrainInput> for CudaTrainIterator {
    fn progress(&self) -> Progress {
        Progress::new(self.produced, self.total_steps)
    }
}

#[derive(Clone)]
pub(super) struct CudaValidationInput {
    tokens: Arc<Vec<u16>>,
    window_count: usize,
}

#[derive(Clone)]
struct CudaValidDataLoader {
    input: CudaValidationInput,
}

impl CudaValidDataLoader {
    fn new(tokens: Vec<u16>, window_count: usize) -> Self {
        Self {
            input: CudaValidationInput {
                tokens: Arc::new(tokens),
                window_count,
            },
        }
    }
}

impl DataLoader<BurnInnerBackend, CudaValidInput> for CudaValidDataLoader {
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<CudaValidInput> + 'a> {
        Box::new(CudaValidIterator {
            input: self.input.clone(),
            produced: false,
        })
    }

    fn num_items(&self) -> usize {
        self.input.window_count
    }

    fn to_device(
        &self,
        _device: &<BurnInnerBackend as BackendTypes>::Device,
    ) -> Arc<dyn DataLoader<BurnInnerBackend, CudaValidInput>> {
        Arc::new(self.clone())
    }

    fn slice(
        &self,
        _start: usize,
        _end: usize,
    ) -> Arc<dyn DataLoader<BurnInnerBackend, CudaValidInput>> {
        Arc::new(self.clone())
    }
}

struct CudaValidIterator {
    input: CudaValidationInput,
    produced: bool,
}

impl Iterator for CudaValidIterator {
    type Item = CudaValidInput;

    fn next(&mut self) -> Option<Self::Item> {
        if self.produced {
            return None;
        }
        self.produced = true;
        Some(Ok(self.input.clone()))
    }
}

impl DataLoaderIterator<CudaValidInput> for CudaValidIterator {
    fn progress(&self) -> Progress {
        Progress::new(usize::from(self.produced), 1)
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

#[derive(Clone)]
pub(super) struct CudaTrainOutput {
    pub(super) source: String,
    pub(super) window_offset: usize,
    pub(super) batch_size: usize,
    pub(super) seq_len: usize,
    pub(super) stats: Arc<TrainStats>,
}

impl burn::train::ItemLazy for CudaTrainOutput {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl Adaptor<CudaTrainOutput> for CudaTrainOutput {
    fn adapt(&self) -> CudaTrainOutput {
        self.clone()
    }
}

#[derive(Clone)]
pub(super) struct CudaValidOutput {
    val_loss: f32,
    eval_elapsed_s: f64,
    window_count: usize,
    completed_steps: usize,
}

impl burn::train::ItemLazy for CudaValidOutput {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl Adaptor<CudaValidOutput> for CudaValidOutput {
    fn adapt(&self) -> CudaValidOutput {
        self.clone()
    }
}

#[derive(Clone, Copy)]
struct TrainMetricSpec {
    name: &'static str,
    unit: Option<&'static str>,
    higher_is_better: bool,
    field: TrainMetricField,
}

metric_fields! {
    TrainMetricField, TRAIN_METRIC_FIELDS, TrainMetricSpec {
        Loss => ("Loss", None, false),
        ForwardMs => ("Forward", Some("ms"), false),
        BackwardEnqueueMs => ("Backward enqueue", Some("ms"), false),
        LossHostWaitMs => ("Loss host wait", Some("ms"), false),
        OptimizerMs => ("Optimizer", Some("ms"), false),
        AuroraMs => ("Aurora", Some("ms"), false),
        KdaClipMs => ("KDA clip", Some("ms"), false),
        AdamMs => ("Adam", Some("ms"), false),
        EmbeddingLookupMs => ("Embedding lookup", Some("ms"), false),
        TokenEmbeddingMs => ("Token embedding", Some("ms"), false),
        FinalNormMs => ("Final norm", Some("ms"), false),
        BlocksMs => ("Blocks", Some("ms"), false),
        GradNorm => ("Grad norm", None, false),
        AdamLr => ("Adam LR", None, false),
        AuroraLr => ("Aurora LR", None, false),
        Tokens => ("Tokens", None, true),
        Logits => ("Logits", None, true),
        Finite => ("Finite", None, true),
        Nonzero => ("Nonzero", None, true),
        UpdateSkipped => ("Update skipped", None, false),
        SkipLossSpike => ("Skip loss spike", None, false),
        SkipGradNormSpike => ("Skip grad norm spike", None, false),
        SkipNonFinite => ("Skip non finite", None, false),
        WindowOffset => ("Window offset", None, true),
        BatchSize => ("Batch size", None, true),
        SeqLen => ("Seq len", None, true),
    }
}

impl TrainMetricField {
    fn value(self, item: &CudaTrainOutput) -> f64 {
        let stats = item.stats.as_ref();
        let bool_value = |value: bool| if value { 1.0 } else { 0.0 };
        match self {
            Self::Loss => stats.loss as f64,
            Self::ForwardMs => stats.forward_ms,
            Self::BackwardEnqueueMs => stats.backward_enqueue_ms,
            Self::LossHostWaitMs => stats.loss_host_wait_ms,
            Self::OptimizerMs => stats.optimizer_ms,
            Self::AuroraMs => stats.optimizer.aurora_ms,
            Self::KdaClipMs => stats.optimizer.kda_clip_ms,
            Self::AdamMs => stats.optimizer.adam_ms,
            Self::EmbeddingLookupMs => stats.optimizer.embedding_lookup_ms,
            Self::TokenEmbeddingMs => stats.optimizer.token_embedding_ms,
            Self::FinalNormMs => stats.optimizer.final_norm_ms,
            Self::BlocksMs => stats.optimizer.blocks_ms,
            Self::GradNorm => stats.optimizer.grad_norm as f64,
            Self::AdamLr => stats.optimizer.adam_lr as f64,
            Self::AuroraLr => stats.optimizer.aurora_lr as f64,
            Self::Tokens => stats.tokens as f64,
            Self::Logits => stats.logits as f64,
            Self::Finite => bool_value(stats.finite),
            Self::Nonzero => bool_value(stats.nonzero),
            Self::UpdateSkipped => bool_value(stats.optimizer.update_skipped),
            Self::SkipLossSpike => bool_value(stats.optimizer.skip_loss_spike),
            Self::SkipGradNormSpike => bool_value(stats.optimizer.skip_grad_norm_spike),
            Self::SkipNonFinite => bool_value(stats.optimizer.skip_non_finite),
            Self::WindowOffset => item.window_offset as f64,
            Self::BatchSize => item.batch_size as f64,
            Self::SeqLen => item.seq_len as f64,
        }
    }
}

#[derive(Clone)]
struct CudaTrainMetric {
    spec: TrainMetricSpec,
    state: MetricAccumulator,
}

impl CudaTrainMetric {
    fn new(spec: TrainMetricSpec) -> Self {
        Self {
            spec,
            state: MetricAccumulator::default(),
        }
    }
}

impl Metric for CudaTrainMetric {
    type Input = CudaTrainOutput;

    fn name(&self) -> MetricName {
        Arc::new(self.spec.name.to_string())
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: self.spec.unit.map(str::to_string),
            higher_is_better: self.spec.higher_is_better,
        }
        .into()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        self.state
            .update(self.spec.field.value(item), self.spec.unit)
    }

    fn clear(&mut self) {
        self.state.clear();
    }
}

impl Numeric for CudaTrainMetric {
    fn value(&self) -> NumericEntry {
        self.state.value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

#[derive(Clone)]
struct CudaSourceMetric {
    name: Arc<String>,
}

impl CudaSourceMetric {
    fn new() -> Self {
        Self {
            name: Arc::new("Source".to_string()),
        }
    }
}

impl Metric for CudaSourceMetric {
    type Input = CudaTrainOutput;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        SerializedEntry {
            formatted: item.source.clone(),
            serialized: item.source.clone(),
        }
    }

    fn clear(&mut self) {}
}

#[derive(Clone, Copy)]
struct ValidMetricSpec {
    name: &'static str,
    unit: Option<&'static str>,
    higher_is_better: bool,
    field: ValidMetricField,
}

metric_fields! {
    ValidMetricField, VALID_METRIC_FIELDS, ValidMetricSpec {
        Loss => ("Validation loss", None, false),
        EvalElapsed => ("Eval elapsed", Some("s"), false),
        WindowCount => ("Val windows", None, true),
        CompletedSteps => ("Completed steps", None, true),
    }
}

impl ValidMetricField {
    fn value(self, item: &CudaValidOutput) -> f64 {
        match self {
            Self::Loss => item.val_loss as f64,
            Self::EvalElapsed => item.eval_elapsed_s,
            Self::WindowCount => item.window_count as f64,
            Self::CompletedSteps => item.completed_steps as f64,
        }
    }
}

#[derive(Clone)]
struct CudaValidMetric {
    spec: ValidMetricSpec,
    state: MetricAccumulator,
}

impl CudaValidMetric {
    fn new(spec: ValidMetricSpec) -> Self {
        Self {
            spec,
            state: MetricAccumulator::default(),
        }
    }
}

impl Metric for CudaValidMetric {
    type Input = CudaValidOutput;

    fn name(&self) -> MetricName {
        Arc::new(self.spec.name.to_string())
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: self.spec.unit.map(str::to_string),
            higher_is_better: self.spec.higher_is_better,
        }
        .into()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        self.state
            .update(self.spec.field.value(item), self.spec.unit)
    }

    fn clear(&mut self) {
        self.state.clear();
    }
}

impl Numeric for CudaValidMetric {
    fn value(&self) -> NumericEntry {
        self.state.value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

#[derive(Clone, Default)]
struct MetricAccumulator {
    current: f64,
    sum: f64,
    count: usize,
}

impl MetricAccumulator {
    fn update(&mut self, value: f64, unit: Option<&str>) -> SerializedEntry {
        self.current = value;
        if value.is_finite() {
            self.sum += value;
            self.count += 1;
        }
        SerializedEntry {
            formatted: format_metric_value(value, unit),
            serialized: value.to_string(),
        }
    }

    fn clear(&mut self) {
        *self = Self::default();
    }

    fn value(&self) -> NumericEntry {
        NumericEntry::Value(self.current)
    }

    fn running_value(&self) -> NumericEntry {
        if self.count == 0 {
            NumericEntry::Value(f64::NAN)
        } else {
            NumericEntry::Aggregated {
                aggregated_value: self.sum / self.count as f64,
                count: self.count,
            }
        }
    }
}

fn format_metric_value(value: f64, unit: Option<&str>) -> String {
    match unit {
        Some(unit) => format!("{value:.6} {unit}"),
        None => format!("{value:.6}"),
    }
}

fn register_cuda_metrics(
    mut training: SupervisedTraining<CudaLearningComponents>,
) -> SupervisedTraining<CudaLearningComponents> {
    training = training.metric_train(CudaSourceMetric::new());
    for field in TRAIN_METRIC_FIELDS {
        training = training.metric_train_numeric(CudaTrainMetric::new(field.spec()));
    }
    for field in VALID_METRIC_FIELDS {
        training = training.metric_valid_numeric(CudaValidMetric::new(field.spec()));
    }
    training
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

#[derive(Clone)]
struct RunOutput {
    dir: PathBuf,
}

impl RunOutput {
    fn new(dataset: &str, label: &str) -> AppResult<Self> {
        let dir = default_run_dir(dataset, label);
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    fn dir(&self) -> &Path {
        &self.dir
    }

    fn path(&self, file_name: &str) -> PathBuf {
        self.dir.join(file_name)
    }

    fn write_info(&self, info: &str) -> AppResult {
        fs::write(self.path("run_info.txt"), info)?;
        Ok(())
    }
}

struct BoxedMetricsRenderer {
    inner: Box<dyn MetricsRenderer>,
    hidden_metric_ids: HashSet<MetricId>,
}

impl BoxedMetricsRenderer {
    fn new(inner: Box<dyn MetricsRenderer>) -> Self {
        Self {
            inner,
            hidden_metric_ids: HashSet::new(),
        }
    }

    fn is_hidden(&self, state: &MetricState) -> bool {
        let metric_id = match state {
            MetricState::Generic(entry) => &entry.metric_id,
            MetricState::Numeric(entry, _) => &entry.metric_id,
        };
        self.hidden_metric_ids.contains(metric_id)
    }
}

impl burn::train::renderer::MetricsRendererTraining for BoxedMetricsRenderer {
    fn update_train(&mut self, state: MetricState) {
        if !self.is_hidden(&state) {
            self.inner.update_train(state);
        }
    }

    fn update_valid(&mut self, state: MetricState) {
        if !self.is_hidden(&state) {
            self.inner.update_valid(state);
        }
    }

    fn render_train(&mut self, item: TrainingProgress, progress_indicators: Vec<ProgressType>) {
        self.inner.render_train(item, progress_indicators);
    }

    fn render_valid(&mut self, item: TrainingProgress, progress_indicators: Vec<ProgressType>) {
        self.inner.render_valid(item, progress_indicators);
    }

    fn on_train_end(
        &mut self,
        summary: Option<burn::train::LearnerSummary>,
    ) -> Result<(), Box<dyn core::error::Error>> {
        self.inner.on_train_end(summary)
    }
}

impl burn::train::renderer::MetricsRendererEvaluation for BoxedMetricsRenderer {
    fn update_test(&mut self, name: EvaluationName, state: MetricState) {
        if !self.is_hidden(&state) {
            self.inner.update_test(name, state);
        }
    }

    fn render_test(&mut self, item: EvaluationProgress, progress_indicators: Vec<ProgressType>) {
        self.inner.render_test(item, progress_indicators);
    }

    fn on_test_end(
        &mut self,
        summary: Option<burn::train::LearnerSummary>,
    ) -> Result<(), Box<dyn core::error::Error>> {
        self.inner.on_test_end(summary)
    }
}

impl MetricsRenderer for BoxedMetricsRenderer {
    fn manual_close(&mut self) {
        self.inner.manual_close();
    }

    fn register_metric(&mut self, definition: MetricDefinition) {
        if hidden_renderer_metric(&definition.name) {
            self.hidden_metric_ids.insert(definition.metric_id);
        } else {
            self.inner.register_metric(definition);
        }
    }
}

fn hidden_renderer_metric(name: &str) -> bool {
    name.starts_with("Diagnostic ")
}

fn default_renderer(interrupter: Interrupter) -> Box<dyn MetricsRenderer> {
    let mode = env_nonempty("TRAIN_RENDERER").unwrap_or_else(|| "auto".to_string());
    let persistent = matches!(mode.as_str(), "tui-persistent" | "persistent")
        || env_bool("TRAIN_RENDERER_PERSIST").unwrap_or(false);
    let wants_tui = matches!(
        mode.as_str(),
        "auto" | "tui" | "tui-persistent" | "persistent"
    );

    if wants_tui && std::io::stdout().is_terminal() {
        let renderer = TuiMetricsRendererWrapper::new(interrupter, None);
        if persistent {
            Box::new(renderer.persistent())
        } else {
            Box::new(renderer)
        }
    } else if matches!(mode.as_str(), "tui" | "tui-persistent" | "persistent") {
        eprintln!("train_renderer_fallback=cli reason=stdout_not_tty requested={mode}");
        Box::new(CliMetricsRenderer::new())
    } else {
        Box::new(CliMetricsRenderer::new())
    }
}

fn load_model_path() -> Option<PathBuf> {
    env_nonempty("TRAIN_LOAD_MODEL").map(PathBuf::from)
}

fn save_model_path(run_output: &RunOutput) -> Option<PathBuf> {
    let value = env_nonempty("TRAIN_SAVE_MODEL")?;
    if value == "1" || value.eq_ignore_ascii_case("true") {
        Some(run_output.path("model.ckpt"))
    } else {
        Some(PathBuf::from(value))
    }
}

fn generate_prompt(dataset: &str, train_elapsed_s: f64) -> Option<String> {
    env_nonempty("TRAIN_GENERATE_PROMPT").or_else(|| {
        (train_elapsed_s >= AUTO_GENERATE_MIN_TRAIN_SECONDS)
            .then(|| default_generate_prompt(dataset).to_string())
    })
}

fn generate_tokens() -> usize {
    env_usize("TRAIN_GENERATE_TOKENS").unwrap_or(128)
}

fn sampling_config() -> SamplingConfig {
    SamplingConfig {
        temperature: env_f32("TRAIN_GENERATE_TEMPERATURE").unwrap_or(0.7),
        top_k: env_usize("TRAIN_GENERATE_TOP_K").unwrap_or(32),
        top_p: env_f32("TRAIN_GENERATE_TOP_P").unwrap_or(0.9),
    }
}

fn write_generated_text(run_output: &RunOutput, text: &str) -> AppResult<PathBuf> {
    let path = run_output.path("generated.txt");
    ensure_parent(&path)?;
    fs::write(&path, text)?;
    Ok(path)
}

fn should_log_step(step: usize, step_cap: usize, log_interval: usize) -> bool {
    step == 0 || step + 1 == step_cap || step % log_interval == 0
}

fn should_eval_step(step: usize, step_cap: usize, eval_interval: Option<usize>) -> bool {
    eval_interval.is_some_and(|interval| step == 0 || step + 1 == step_cap || step % interval == 0)
}

fn build_run_info(dataset: &str, config: &TrainConfig) -> String {
    let mut info = String::new();
    push_info(&mut info, "dataset", dataset);
    push_info(&mut info, "training_launcher", "burn");
    push_info(&mut info, "metric_logger", "burn_file");
    push_info(&mut info, "tokenizer", llama2_tokenizer::TOKENIZER_NAME);
    push_info(&mut info, "vocab_size", llama2_tokenizer::VOCAB_SIZE);
    push_info(&mut info, "gpt2_seq_len", GPT2_SEQ_LEN);
    push_info(&mut info, "gpt2_batch_size", GPT2_BATCH_SIZE);
    push_info(&mut info, "gpt2_token_rows", GPT2_TOKEN_ROWS);
    push_info(&mut info, "gpt2_n_layer", GPT2_N_LAYER);
    push_info(&mut info, "gpt2_n_head", GPT2_N_HEAD);
    push_info(&mut info, "gpt2_n_embd", GPT2_N_EMBD);
    push_info(
        &mut info,
        "aurora_cooperative_blocks",
        AURORA_COOPERATIVE_BLOCKS,
    );
    push_info(&mut info, "aurora_matrix_phases", AURORA_MATRIX_PHASES);
    push_info(&mut info, "step_cap", config.step_cap);
    push_info(&mut info, "log_interval", config.log_interval);
    push_info(&mut info, "max_seconds", config.max_seconds);
    push_info(
        &mut info,
        "eval_interval",
        config
            .eval_interval
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
    );
    push_info(&mut info, "seed", format!("{:#x}", config.seed));
    push_run_env(&mut info);
    info
}

fn push_run_env(info: &mut String) {
    for name in [
        "CUDA_DEVICE_INDEX",
        "TRAIN_DATASET",
        "TRAIN_LOAD_MODEL",
        "TRAIN_SAVE_MODEL",
        "TRAIN_STEPS",
        "TRAIN_LOG_INTERVAL",
        "TRAIN_EVAL_INTERVAL",
        "TRAIN_MAX_SECONDS",
        "TRAIN_REPEAT_BATCH",
        "TRAIN_SEED",
        "TRAIN_LR_SCALE",
        "TRAIN_ADAM_LR_SCALE",
        "TRAIN_NEXTLAT_LR_SCALE",
        "TRAIN_LR_WARMUP_STEPS",
        "TRAIN_LR_START_RATIO",
        "TRAIN_AMUSE_BETA1",
        "TRAIN_AMUSE_RHO",
        "TRAIN_SKIP_UNSTABLE_UPDATES",
        "TRAIN_SKIP_ROLLING_INTERVAL",
        "TRAIN_SKIP_SIGMA_FACTOR",
        "TRAIN_SKIP_USE_LOSS",
        "TRAIN_SKIP_USE_GRAD_NORM",
        "TRAIN_GENERATE_PROMPT",
        "TRAIN_GENERATE_TOKENS",
        "TRAIN_GENERATE_TEMPERATURE",
        "TRAIN_GENERATE_TOP_K",
        "TRAIN_GENERATE_TOP_P",
    ] {
        if let Ok(value) = std::env::var(name) {
            push_info(info, name, value);
        }
    }
}

fn push_info(info: &mut String, name: &str, value: impl std::fmt::Display) {
    use std::fmt::Write;
    let _ = writeln!(info, "{name}={value}");
}

fn default_run_dir(dataset: &str, label: &str) -> PathBuf {
    PathBuf::from(RUNS_DIR).join(format!(
        "{}_{}_{}",
        utc_stamp(),
        sanitize_path_part(dataset),
        sanitize_path_part(label)
    ))
}

fn utc_stamp() -> String {
    let now = OffsetDateTime::now_utc();
    format!(
        "{:04}{:02}{:02}_{:02}{:02}{:02}Z",
        now.year(),
        u8::from(now.month()),
        now.day(),
        now.hour(),
        now.minute(),
        now.second()
    )
}

fn sanitize_path_part(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn ensure_parent(path: &Path) -> AppResult {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn default_generate_prompt(dataset: &str) -> &'static str {
    match dataset {
        "shakespeare" => DEFAULT_SHAKESPEARE_PROMPT,
        _ => DEFAULT_SYNTH_PROMPT,
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

fn env_u64(name: &str) -> Option<u64> {
    let value = std::env::var(name).ok()?;
    value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
        .map(|hex| u64::from_str_radix(hex, 16).ok())
        .unwrap_or_else(|| value.parse().ok())
}

fn env_bool(name: &str) -> Option<bool> {
    let value = std::env::var(name).ok()?;
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}
