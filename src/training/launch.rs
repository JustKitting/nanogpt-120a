use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use burn::data::dataloader::Progress;
use burn::train::Interrupter;
use burn::train::logger::{FileMetricLogger, MetricLogger};
use burn::train::metric::store::{EpochSummary, MetricsUpdate, NumericMetricUpdate, Split};
use burn::train::metric::{
    MetricAttributes, MetricDefinition, MetricEntry, MetricId, NumericAttributes, NumericEntry,
    SerializedEntry,
};
use burn::train::renderer::tui::TuiMetricsRendererWrapper;
use burn::train::renderer::{
    CliMetricsRenderer, MetricState, MetricsRenderer, ProgressType, TrainingProgress,
};
use gpt2_nvfp4::{
    GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_N_LAYER, GPT2_SEQ_LEN, GPT2_TOKEN_ROWS,
};
use rust_kernels_cuda::optimizer::{AURORA_COOPERATIVE_BLOCKS, AURORA_MATRIX_PHASES};
use time::OffsetDateTime;

use super::{SamplingConfig, TokenDataLoader, TrainStats, Trainer};
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
    renderer: Box<dyn MetricsRenderer>,
    interrupter: Interrupter,
}

impl BurnTrainingLauncher {
    fn new(dataset: String, config: TrainConfig) -> Self {
        let interrupter = Interrupter::new();
        Self {
            dataset,
            config,
            renderer: default_renderer(interrupter.clone()),
            interrupter,
        }
    }

    fn run(mut self) -> AppResult {
        let mut trainer = Trainer::new(self.config.seed)?;
        let run_label = format!("{}s", self.config.max_seconds.round() as u64);
        let run_output = RunOutput::new(&self.dataset, &run_label)?;
        let mut metrics = BurnMetrics::register(self.renderer.as_mut(), run_output.path("metrics"));
        println!("run_dir={}", run_output.dir().display());
        println!("metrics_dir={}", metrics.dir().display());

        if let Some(path) = load_model_path() {
            trainer.load_model(&path)?;
            println!("loaded_model={}", path.display());
        }

        let mut data = TokenDataLoader::from_training_dataset()?;
        let mut train_batch = trainer.reusable_default_batch()?;
        let wall_clock = WallClockBudget::new(self.config.max_seconds);
        let validation_tokens = data.validation_tokens()?;
        let validation_window_count = TokenDataLoader::validation_window_count();

        run_output.write_info(&build_run_info(&self.dataset, &self.config))?;
        println!(
            "training_tokens={} max_seconds={:.3} step_cap={}",
            data.token_count(),
            self.config.max_seconds,
            self.config.step_cap
        );

        let mut completed_steps = 0usize;
        for step in 0..self.config.step_cap {
            let log_step = should_log_step(step, self.config.step_cap, self.config.log_interval);
            let window = data.next_batch()?;
            let source = window.source.display().to_string();
            let batch = trainer.upload_default_batch(&mut train_batch, &window.tokens)?;
            let stats = trainer.train_step(&batch, log_step)?;
            completed_steps = step + 1;

            if log_step {
                self.render_train_step(
                    &mut metrics,
                    step,
                    completed_steps,
                    &source,
                    window.offset,
                    window.batch_size,
                    window.seq_len,
                    &stats,
                );
            }
            if should_eval_step(step, self.config.step_cap, self.config.eval_interval) {
                let eval_start = Instant::now();
                let val_loss =
                    trainer.eval_loss_windows(&validation_tokens, validation_window_count)?;
                self.render_validation(
                    &mut metrics,
                    step,
                    completed_steps,
                    validation_window_count,
                    val_loss,
                    eval_start.elapsed().as_secs_f64(),
                );
            }
            if self.interrupter.should_stop() {
                println!(
                    "stopped_by_burn_interrupter=true elapsed_s={:.3} completed_steps={completed_steps}",
                    wall_clock.elapsed_seconds(),
                );
                break;
            }
            if wall_clock.expired() {
                println!(
                    "stopped_by_wall_clock=true elapsed_s={:.3} completed_steps={}",
                    wall_clock.elapsed_seconds(),
                    step + 1,
                );
                break;
            }
        }
        let train_elapsed_s = wall_clock.elapsed_seconds();

        let eval_start = Instant::now();
        let val_loss = trainer.eval_loss_windows(&validation_tokens, validation_window_count)?;
        let eval_elapsed_s = eval_start.elapsed().as_secs_f64();
        self.render_validation(
            &mut metrics,
            completed_steps.saturating_sub(1),
            completed_steps,
            validation_window_count,
            val_loss,
            eval_elapsed_s,
        );
        metrics.finish();
        println!(
            "heldout_eval split=val val_loss={val_loss:.6} train_elapsed_s={train_elapsed_s:.3} eval_elapsed_s={eval_elapsed_s:.3} completed_steps={completed_steps}",
        );

        if let Some(path) = save_model_path(&run_output) {
            ensure_parent(&path)?;
            trainer.save_model(&path)?;
            println!("saved_model={}", path.display());
        }

        if let Some(prompt) = generate_prompt(&self.dataset, train_elapsed_s) {
            let text = trainer.generate_sampled(&prompt, generate_tokens(), sampling_config())?;
            let generated_path = write_generated_text(&run_output, &text)?;
            println!("generated_text={}", generated_path.display());
            println!("generated_text_begin");
            println!("{text}");
            println!("generated_text_end");
        }

        self.renderer.on_train_end(None).ok();
        Ok(())
    }

    fn render_train_step(
        &mut self,
        metrics: &mut BurnMetrics,
        step: usize,
        completed_steps: usize,
        source: &str,
        window_offset: usize,
        batch_size: usize,
        seq_len: usize,
        stats: &TrainStats,
    ) {
        for (metric, value) in metrics.train_render_values(stats) {
            self.renderer.update_train(metric_state(&metric, value));
        }
        metrics.log_train(metrics.train_log_values(
            step,
            source,
            window_offset,
            batch_size,
            seq_len,
            stats,
        ));
        self.renderer.render_train(
            training_progress(step, completed_steps, self.config.step_cap),
            vec![
                ProgressType::Value {
                    tag: "Batch".to_string(),
                    value: batch_size,
                },
                ProgressType::Value {
                    tag: "Seq".to_string(),
                    value: seq_len,
                },
            ],
        );
    }

    fn render_validation(
        &mut self,
        metrics: &mut BurnMetrics,
        step: usize,
        completed_steps: usize,
        validation_window_count: usize,
        val_loss: f32,
        eval_elapsed_s: f64,
    ) {
        for (metric, value) in metrics.valid_values(val_loss, eval_elapsed_s) {
            self.renderer.update_valid(metric_state(&metric, value));
        }
        metrics.log_valid(metrics.valid_values(val_loss, eval_elapsed_s));
        self.renderer.render_valid(
            TrainingProgress {
                progress: Some(Progress::new(
                    validation_window_count,
                    validation_window_count,
                )),
                global_progress: Progress::new(completed_steps, self.config.step_cap),
                iteration: Some(step),
            },
            vec![ProgressType::Value {
                tag: "Val windows".to_string(),
                value: validation_window_count,
            }],
        );
    }
}

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

struct BurnMetrics {
    directory: PathBuf,
    logger: FileMetricLogger,
    loss: BurnMetric,
    forward_ms: BurnMetric,
    backward_enqueue_ms: BurnMetric,
    loss_host_wait_ms: BurnMetric,
    optimizer_ms: BurnMetric,
    aurora_ms: BurnMetric,
    kda_clip_ms: BurnMetric,
    adam_ms: BurnMetric,
    embedding_lookup_ms: BurnMetric,
    token_embedding_ms: BurnMetric,
    final_norm_ms: BurnMetric,
    blocks_ms: BurnMetric,
    grad_norm: BurnMetric,
    adam_lr: BurnMetric,
    aurora_lr: BurnMetric,
    tokens: BurnMetric,
    logits: BurnMetric,
    finite: BurnMetric,
    nonzero: BurnMetric,
    update_skipped: BurnMetric,
    skip_loss_spike: BurnMetric,
    skip_grad_norm_spike: BurnMetric,
    skip_non_finite: BurnMetric,
    window_offset: BurnMetric,
    batch_size: BurnMetric,
    seq_len: BurnMetric,
    source: BurnMetric,
    diagnostic_step: BurnMetric,
    diagnostic_update_count: BurnMetric,
    diagnostic_positive_update_dot: BurnMetric,
    diagnostic_zero_grad_changed: BurnMetric,
    diagnostic_max_update_to_weight_rms: BurnMetric,
    diagnostic_dlogits_rms: BurnMetric,
    diagnostic_dlogits_max: BurnMetric,
    diagnostic_d_lm_head_rms: BurnMetric,
    diagnostic_d_lm_head_max: BurnMetric,
    diagnostic_d_embedding_rms: BurnMetric,
    diagnostic_d_embedding_max: BurnMetric,
    diagnostic_token_embedding_global_before: BurnMetric,
    diagnostic_token_embedding_global_after: BurnMetric,
    diagnostic_token_embedding_changed_bytes: BurnMetric,
    diagnostic_token_embedding_hash_before: BurnMetric,
    diagnostic_token_embedding_hash_after: BurnMetric,
    diagnostic_tensor_index: BurnMetric,
    diagnostic_tensor_name: BurnMetric,
    diagnostic_tensor_len: BurnMetric,
    diagnostic_tensor_grad_rms: BurnMetric,
    diagnostic_tensor_grad_max: BurnMetric,
    diagnostic_tensor_grad_nonzero: BurnMetric,
    diagnostic_tensor_grad_finite: BurnMetric,
    diagnostic_tensor_weight_rms_before: BurnMetric,
    diagnostic_tensor_weight_rms_after: BurnMetric,
    diagnostic_tensor_delta_rms: BurnMetric,
    diagnostic_tensor_delta_max: BurnMetric,
    diagnostic_tensor_update_to_weight_rms: BurnMetric,
    diagnostic_tensor_delta_grad_dot: BurnMetric,
    diagnostic_tensor_delta_grad_cos: BurnMetric,
    diagnostic_tensor_predicted_delta_rms: BurnMetric,
    diagnostic_tensor_predicted_delta_grad_dot: BurnMetric,
    diagnostic_tensor_predicted_delta_grad_cos: BurnMetric,
    diagnostic_tensor_quant_error_rms: BurnMetric,
    diagnostic_tensor_quant_error_to_predicted_delta_rms: BurnMetric,
    diagnostic_tensor_changed_bytes: BurnMetric,
    diagnostic_tensor_changed_scales: BurnMetric,
    diagnostic_tensor_global_before: BurnMetric,
    diagnostic_tensor_global_after: BurnMetric,
    val_loss: BurnMetric,
    eval_elapsed_s: BurnMetric,
}

impl BurnMetrics {
    fn register(renderer: &mut dyn MetricsRenderer, directory: PathBuf) -> Self {
        let mut logger = FileMetricLogger::new(&directory);
        Self {
            directory,
            loss: register_numeric(renderer, &mut logger, "train_loss", "Loss", None, false),
            forward_ms: register_numeric(
                renderer,
                &mut logger,
                "forward_ms",
                "Forward",
                Some("ms"),
                false,
            ),
            backward_enqueue_ms: register_numeric(
                renderer,
                &mut logger,
                "backward_enqueue_ms",
                "Backward enqueue",
                Some("ms"),
                false,
            ),
            loss_host_wait_ms: register_numeric(
                renderer,
                &mut logger,
                "loss_host_wait_ms",
                "Loss host wait",
                Some("ms"),
                false,
            ),
            optimizer_ms: register_numeric(
                renderer,
                &mut logger,
                "optimizer_ms",
                "Optimizer",
                Some("ms"),
                false,
            ),
            aurora_ms: register_numeric(
                renderer,
                &mut logger,
                "aurora_ms",
                "Aurora",
                Some("ms"),
                false,
            ),
            kda_clip_ms: register_numeric(
                renderer,
                &mut logger,
                "kda_clip_ms",
                "KDA clip",
                Some("ms"),
                false,
            ),
            adam_ms: register_numeric(renderer, &mut logger, "adam_ms", "Adam", Some("ms"), false),
            embedding_lookup_ms: register_numeric(
                renderer,
                &mut logger,
                "embedding_lookup_ms",
                "Embedding lookup",
                Some("ms"),
                false,
            ),
            token_embedding_ms: register_numeric(
                renderer,
                &mut logger,
                "token_embedding_ms",
                "Token embedding",
                Some("ms"),
                false,
            ),
            final_norm_ms: register_numeric(
                renderer,
                &mut logger,
                "final_norm_ms",
                "Final norm",
                Some("ms"),
                false,
            ),
            blocks_ms: register_numeric(
                renderer,
                &mut logger,
                "blocks_ms",
                "Blocks",
                Some("ms"),
                false,
            ),
            grad_norm: register_numeric(
                renderer,
                &mut logger,
                "grad_norm",
                "Grad norm",
                None,
                false,
            ),
            adam_lr: register_numeric(renderer, &mut logger, "adam_lr", "Adam LR", None, false),
            aurora_lr: register_numeric(
                renderer,
                &mut logger,
                "aurora_lr",
                "Aurora LR",
                None,
                false,
            ),
            tokens: register_numeric(renderer, &mut logger, "tokens", "Tokens", None, true),
            logits: register_numeric(renderer, &mut logger, "logits", "Logits", None, true),
            finite: register_numeric(renderer, &mut logger, "finite", "Finite", None, true),
            nonzero: register_numeric(renderer, &mut logger, "nonzero", "Nonzero", None, true),
            update_skipped: register_numeric(
                renderer,
                &mut logger,
                "update_skipped",
                "Update skipped",
                None,
                false,
            ),
            skip_loss_spike: register_numeric(
                renderer,
                &mut logger,
                "skip_loss_spike",
                "Skip loss spike",
                None,
                false,
            ),
            skip_grad_norm_spike: register_numeric(
                renderer,
                &mut logger,
                "skip_grad_norm_spike",
                "Skip grad norm spike",
                None,
                false,
            ),
            skip_non_finite: register_numeric(
                renderer,
                &mut logger,
                "skip_non_finite",
                "Skip non finite",
                None,
                false,
            ),
            window_offset: register_numeric(
                renderer,
                &mut logger,
                "window_offset",
                "Window offset",
                None,
                true,
            ),
            batch_size: register_numeric(
                renderer,
                &mut logger,
                "batch_size",
                "Batch size",
                None,
                true,
            ),
            seq_len: register_numeric(renderer, &mut logger, "seq_len", "Seq len", None, true),
            source: register_generic(renderer, &mut logger, "source", "Source"),
            diagnostic_step: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_step",
                "Diagnostic step",
                None,
                true,
            ),
            diagnostic_update_count: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_update_count",
                "Diagnostic update count",
                None,
                true,
            ),
            diagnostic_positive_update_dot: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_positive_update_dot",
                "Diagnostic positive update dot",
                None,
                true,
            ),
            diagnostic_zero_grad_changed: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_zero_grad_changed",
                "Diagnostic zero grad changed",
                None,
                false,
            ),
            diagnostic_max_update_to_weight_rms: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_max_update_to_weight_rms",
                "Diagnostic max update to weight RMS",
                None,
                false,
            ),
            diagnostic_dlogits_rms: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_dlogits_rms",
                "Diagnostic dlogits RMS",
                None,
                false,
            ),
            diagnostic_dlogits_max: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_dlogits_max",
                "Diagnostic dlogits max",
                None,
                false,
            ),
            diagnostic_d_lm_head_rms: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_d_lm_head_rms",
                "Diagnostic d lm head RMS",
                None,
                false,
            ),
            diagnostic_d_lm_head_max: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_d_lm_head_max",
                "Diagnostic d lm head max",
                None,
                false,
            ),
            diagnostic_d_embedding_rms: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_d_embedding_rms",
                "Diagnostic d embedding RMS",
                None,
                false,
            ),
            diagnostic_d_embedding_max: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_d_embedding_max",
                "Diagnostic d embedding max",
                None,
                false,
            ),
            diagnostic_token_embedding_global_before: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_token_embedding_global_before",
                "Diagnostic token embedding global before",
                None,
                false,
            ),
            diagnostic_token_embedding_global_after: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_token_embedding_global_after",
                "Diagnostic token embedding global after",
                None,
                false,
            ),
            diagnostic_token_embedding_changed_bytes: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_token_embedding_changed_bytes",
                "Diagnostic token embedding changed bytes",
                None,
                true,
            ),
            diagnostic_token_embedding_hash_before: register_generic(
                renderer,
                &mut logger,
                "diagnostic_token_embedding_hash_before",
                "Diagnostic token embedding hash before",
            ),
            diagnostic_token_embedding_hash_after: register_generic(
                renderer,
                &mut logger,
                "diagnostic_token_embedding_hash_after",
                "Diagnostic token embedding hash after",
            ),
            diagnostic_tensor_index: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_index",
                "Diagnostic tensor index",
                None,
                true,
            ),
            diagnostic_tensor_name: register_generic(
                renderer,
                &mut logger,
                "diagnostic_tensor_name",
                "Diagnostic tensor name",
            ),
            diagnostic_tensor_len: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_len",
                "Diagnostic tensor len",
                None,
                true,
            ),
            diagnostic_tensor_grad_rms: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_grad_rms",
                "Diagnostic tensor grad RMS",
                None,
                false,
            ),
            diagnostic_tensor_grad_max: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_grad_max",
                "Diagnostic tensor grad max",
                None,
                false,
            ),
            diagnostic_tensor_grad_nonzero: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_grad_nonzero",
                "Diagnostic tensor grad nonzero",
                None,
                true,
            ),
            diagnostic_tensor_grad_finite: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_grad_finite",
                "Diagnostic tensor grad finite",
                None,
                true,
            ),
            diagnostic_tensor_weight_rms_before: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_weight_rms_before",
                "Diagnostic tensor weight RMS before",
                None,
                false,
            ),
            diagnostic_tensor_weight_rms_after: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_weight_rms_after",
                "Diagnostic tensor weight RMS after",
                None,
                false,
            ),
            diagnostic_tensor_delta_rms: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_delta_rms",
                "Diagnostic tensor delta RMS",
                None,
                false,
            ),
            diagnostic_tensor_delta_max: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_delta_max",
                "Diagnostic tensor delta max",
                None,
                false,
            ),
            diagnostic_tensor_update_to_weight_rms: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_update_to_weight_rms",
                "Diagnostic tensor update to weight RMS",
                None,
                false,
            ),
            diagnostic_tensor_delta_grad_dot: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_delta_grad_dot",
                "Diagnostic tensor delta grad dot",
                None,
                false,
            ),
            diagnostic_tensor_delta_grad_cos: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_delta_grad_cos",
                "Diagnostic tensor delta grad cos",
                None,
                false,
            ),
            diagnostic_tensor_predicted_delta_rms: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_predicted_delta_rms",
                "Diagnostic tensor predicted delta RMS",
                None,
                false,
            ),
            diagnostic_tensor_predicted_delta_grad_dot: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_predicted_delta_grad_dot",
                "Diagnostic tensor predicted delta grad dot",
                None,
                false,
            ),
            diagnostic_tensor_predicted_delta_grad_cos: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_predicted_delta_grad_cos",
                "Diagnostic tensor predicted delta grad cos",
                None,
                false,
            ),
            diagnostic_tensor_quant_error_rms: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_quant_error_rms",
                "Diagnostic tensor quant error RMS",
                None,
                false,
            ),
            diagnostic_tensor_quant_error_to_predicted_delta_rms: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_quant_error_to_predicted_delta_rms",
                "Diagnostic tensor quant error to predicted delta RMS",
                None,
                false,
            ),
            diagnostic_tensor_changed_bytes: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_changed_bytes",
                "Diagnostic tensor changed bytes",
                None,
                true,
            ),
            diagnostic_tensor_changed_scales: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_changed_scales",
                "Diagnostic tensor changed scales",
                None,
                true,
            ),
            diagnostic_tensor_global_before: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_global_before",
                "Diagnostic tensor global before",
                None,
                false,
            ),
            diagnostic_tensor_global_after: register_numeric(
                renderer,
                &mut logger,
                "diagnostic_tensor_global_after",
                "Diagnostic tensor global after",
                None,
                false,
            ),
            val_loss: register_numeric(
                renderer,
                &mut logger,
                "val_loss",
                "Validation loss",
                None,
                false,
            ),
            eval_elapsed_s: register_numeric(
                renderer,
                &mut logger,
                "eval_elapsed_s",
                "Eval elapsed",
                Some("s"),
                false,
            ),
            logger,
        }
    }

    fn dir(&self) -> &Path {
        &self.directory
    }

    fn train_render_values(&self, stats: &TrainStats) -> Vec<(BurnMetric, f64)> {
        vec![
            (self.loss.clone(), stats.loss as f64),
            (self.forward_ms.clone(), stats.forward_ms),
            (self.backward_enqueue_ms.clone(), stats.backward_enqueue_ms),
            (self.optimizer_ms.clone(), stats.optimizer_ms),
            (self.grad_norm.clone(), stats.optimizer.grad_norm as f64),
        ]
    }

    fn train_log_values(
        &self,
        step: usize,
        source: &str,
        window_offset: usize,
        batch_size: usize,
        seq_len: usize,
        stats: &TrainStats,
    ) -> Vec<BurnLogValue> {
        let bool_value = |value: bool| if value { 1.0 } else { 0.0 };
        let mut values = vec![
            generic_value(self.source.clone(), source),
            numeric_value(self.loss.clone(), stats.loss as f64),
            numeric_value(self.forward_ms.clone(), stats.forward_ms),
            numeric_value(self.backward_enqueue_ms.clone(), stats.backward_enqueue_ms),
            numeric_value(self.loss_host_wait_ms.clone(), stats.loss_host_wait_ms),
            numeric_value(self.optimizer_ms.clone(), stats.optimizer_ms),
            numeric_value(self.aurora_ms.clone(), stats.optimizer.aurora_ms),
            numeric_value(self.kda_clip_ms.clone(), stats.optimizer.kda_clip_ms),
            numeric_value(self.adam_ms.clone(), stats.optimizer.adam_ms),
            numeric_value(
                self.embedding_lookup_ms.clone(),
                stats.optimizer.embedding_lookup_ms,
            ),
            numeric_value(
                self.token_embedding_ms.clone(),
                stats.optimizer.token_embedding_ms,
            ),
            numeric_value(self.final_norm_ms.clone(), stats.optimizer.final_norm_ms),
            numeric_value(self.blocks_ms.clone(), stats.optimizer.blocks_ms),
            numeric_value(self.grad_norm.clone(), stats.optimizer.grad_norm as f64),
            numeric_value(self.adam_lr.clone(), stats.optimizer.adam_lr as f64),
            numeric_value(self.aurora_lr.clone(), stats.optimizer.aurora_lr as f64),
            numeric_value(self.tokens.clone(), stats.tokens as f64),
            numeric_value(self.logits.clone(), stats.logits as f64),
            numeric_value(self.finite.clone(), bool_value(stats.finite)),
            numeric_value(self.nonzero.clone(), bool_value(stats.nonzero)),
            numeric_value(
                self.update_skipped.clone(),
                bool_value(stats.optimizer.update_skipped),
            ),
            numeric_value(
                self.skip_loss_spike.clone(),
                bool_value(stats.optimizer.skip_loss_spike),
            ),
            numeric_value(
                self.skip_grad_norm_spike.clone(),
                bool_value(stats.optimizer.skip_grad_norm_spike),
            ),
            numeric_value(
                self.skip_non_finite.clone(),
                bool_value(stats.optimizer.skip_non_finite),
            ),
            numeric_value(self.window_offset.clone(), window_offset as f64),
            numeric_value(self.batch_size.clone(), batch_size as f64),
            numeric_value(self.seq_len.clone(), seq_len as f64),
        ];

        if let Some(trace) = &stats.diagnostics {
            values.extend([
                numeric_value(self.diagnostic_step.clone(), step as f64),
                numeric_value(
                    self.diagnostic_update_count.clone(),
                    trace.update_count as f64,
                ),
                numeric_value(
                    self.diagnostic_positive_update_dot.clone(),
                    trace.positive_update_dot_count as f64,
                ),
                numeric_value(
                    self.diagnostic_zero_grad_changed.clone(),
                    trace.zero_grad_changed_count as f64,
                ),
                numeric_value(
                    self.diagnostic_max_update_to_weight_rms.clone(),
                    trace.max_update_to_weight_rms as f64,
                ),
                numeric_value(
                    self.diagnostic_dlogits_rms.clone(),
                    trace.dlogits_rms as f64,
                ),
                numeric_value(
                    self.diagnostic_dlogits_max.clone(),
                    trace.dlogits_max as f64,
                ),
                numeric_value(
                    self.diagnostic_d_lm_head_rms.clone(),
                    trace.d_lm_head_rms as f64,
                ),
                numeric_value(
                    self.diagnostic_d_lm_head_max.clone(),
                    trace.d_lm_head_max as f64,
                ),
                numeric_value(
                    self.diagnostic_d_embedding_rms.clone(),
                    trace.d_embedding_rms as f64,
                ),
                numeric_value(
                    self.diagnostic_d_embedding_max.clone(),
                    trace.d_embedding_max as f64,
                ),
                numeric_value(
                    self.diagnostic_token_embedding_global_before.clone(),
                    trace.token_embedding_global_before as f64,
                ),
                numeric_value(
                    self.diagnostic_token_embedding_global_after.clone(),
                    trace.token_embedding_global_after as f64,
                ),
                numeric_value(
                    self.diagnostic_token_embedding_changed_bytes.clone(),
                    trace.token_embedding_changed_bytes as f64,
                ),
                generic_value(
                    self.diagnostic_token_embedding_hash_before.clone(),
                    format!("{:016x}", trace.token_embedding_hash_before),
                ),
                generic_value(
                    self.diagnostic_token_embedding_hash_after.clone(),
                    format!("{:016x}", trace.token_embedding_hash_after),
                ),
            ]);

            for (index, update) in trace.updates.iter().enumerate() {
                values.extend([
                    numeric_value(self.diagnostic_step.clone(), step as f64),
                    numeric_value(self.diagnostic_tensor_index.clone(), index as f64),
                    generic_value(self.diagnostic_tensor_name.clone(), update.name.as_str()),
                    numeric_value(self.diagnostic_tensor_len.clone(), update.len as f64),
                    numeric_value(
                        self.diagnostic_tensor_grad_rms.clone(),
                        update.grad_rms as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_grad_max.clone(),
                        update.grad_max as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_grad_nonzero.clone(),
                        update.grad_nonzero as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_grad_finite.clone(),
                        bool_value(update.grad_finite),
                    ),
                    numeric_value(
                        self.diagnostic_tensor_weight_rms_before.clone(),
                        update.weight_rms_before as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_weight_rms_after.clone(),
                        update.weight_rms_after as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_delta_rms.clone(),
                        update.delta_rms as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_delta_max.clone(),
                        update.delta_max as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_update_to_weight_rms.clone(),
                        update.update_to_weight_rms as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_delta_grad_dot.clone(),
                        update.delta_grad_dot as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_delta_grad_cos.clone(),
                        update.delta_grad_cos as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_predicted_delta_rms.clone(),
                        update.predicted_delta_rms as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_predicted_delta_grad_dot.clone(),
                        update.predicted_delta_grad_dot as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_predicted_delta_grad_cos.clone(),
                        update.predicted_delta_grad_cos as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_quant_error_rms.clone(),
                        update.quant_error_rms as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_quant_error_to_predicted_delta_rms
                            .clone(),
                        update.quant_error_to_predicted_delta_rms as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_changed_bytes.clone(),
                        update.changed_bytes as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_changed_scales.clone(),
                        update.changed_scales as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_global_before.clone(),
                        update.global_before as f64,
                    ),
                    numeric_value(
                        self.diagnostic_tensor_global_after.clone(),
                        update.global_after as f64,
                    ),
                ]);
            }
        }

        values
    }

    fn valid_values(&self, val_loss: f32, eval_elapsed_s: f64) -> Vec<(BurnMetric, f64)> {
        vec![
            (self.val_loss.clone(), val_loss as f64),
            (self.eval_elapsed_s.clone(), eval_elapsed_s),
        ]
    }

    fn log_train(&mut self, values: Vec<BurnLogValue>) {
        self.logger
            .log(metrics_update(values), TRAIN_EPOCH, &Split::Train);
    }

    fn log_valid(&mut self, values: Vec<(BurnMetric, f64)>) {
        self.logger.log(
            metrics_update(numeric_values(values)),
            TRAIN_EPOCH,
            &Split::Valid,
        );
    }

    fn finish(&mut self) {
        self.logger
            .log_epoch_summary(EpochSummary::new(TRAIN_EPOCH, Split::Train));
        self.logger
            .log_epoch_summary(EpochSummary::new(TRAIN_EPOCH, Split::Valid));
    }
}

#[derive(Clone)]
struct BurnMetric {
    id: MetricId,
}

enum BurnLogValue {
    Numeric(BurnMetric, f64),
    Generic(BurnMetric, String),
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

fn register_numeric(
    renderer: &mut dyn MetricsRenderer,
    logger: &mut dyn MetricLogger,
    id: &str,
    name: &str,
    unit: Option<&str>,
    higher_is_better: bool,
) -> BurnMetric {
    let id = MetricId::new(Arc::new(id.to_string()));
    let definition = MetricDefinition {
        metric_id: id.clone(),
        name: name.to_string(),
        description: None,
        attributes: MetricAttributes::Numeric(NumericAttributes {
            unit: unit.map(str::to_string),
            higher_is_better,
        }),
    };
    renderer.register_metric(definition.clone());
    logger.log_metric_definition(definition);
    BurnMetric { id }
}

fn register_generic(
    renderer: &mut dyn MetricsRenderer,
    logger: &mut dyn MetricLogger,
    id: &str,
    name: &str,
) -> BurnMetric {
    let id = MetricId::new(Arc::new(id.to_string()));
    let definition = MetricDefinition {
        metric_id: id.clone(),
        name: name.to_string(),
        description: None,
        attributes: MetricAttributes::None,
    };
    renderer.register_metric(definition.clone());
    logger.log_metric_definition(definition);
    BurnMetric { id }
}

fn metric_state(metric: &BurnMetric, value: f64) -> MetricState {
    MetricState::Numeric(metric_entry(metric, value), NumericEntry::Value(value))
}

fn numeric_value(metric: BurnMetric, value: f64) -> BurnLogValue {
    BurnLogValue::Numeric(metric, value)
}

fn generic_value(metric: BurnMetric, value: impl Into<String>) -> BurnLogValue {
    BurnLogValue::Generic(metric, value.into())
}

fn numeric_values(values: Vec<(BurnMetric, f64)>) -> Vec<BurnLogValue> {
    values
        .into_iter()
        .map(|(metric, value)| numeric_value(metric, value))
        .collect()
}

fn metrics_update(values: Vec<BurnLogValue>) -> MetricsUpdate {
    let mut entries = Vec::new();
    let mut entries_numeric = Vec::new();

    for value in values {
        match value {
            BurnLogValue::Numeric(metric, value) => {
                let numeric = NumericEntry::Value(value);
                entries_numeric.push(NumericMetricUpdate::new(
                    metric_entry(&metric, value),
                    numeric.clone(),
                    numeric,
                ));
            }
            BurnLogValue::Generic(metric, value) => {
                entries.push(generic_metric_entry(&metric, value));
            }
        }
    }

    MetricsUpdate::new(entries, entries_numeric)
}

fn metric_entry(metric: &BurnMetric, value: f64) -> MetricEntry {
    MetricEntry::new(
        metric.id.clone(),
        SerializedEntry {
            formatted: format!("{value:.6}"),
            serialized: value.to_string(),
        },
    )
}

fn generic_metric_entry(metric: &BurnMetric, value: String) -> MetricEntry {
    MetricEntry::new(
        metric.id.clone(),
        SerializedEntry {
            formatted: value.clone(),
            serialized: value,
        },
    )
}

fn training_progress(step: usize, completed_steps: usize, step_cap: usize) -> TrainingProgress {
    TrainingProgress {
        progress: Some(Progress::new(completed_steps, step_cap)),
        global_progress: Progress::new(completed_steps, step_cap),
        iteration: Some(step),
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
