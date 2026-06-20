use std::path::PathBuf;

use crate::training::SamplingConfig;

use super::run_output::RunOutput;

pub const DEFAULT_SEED: u64 = 0x4750_5432;
const DEFAULT_TRAIN_MAX_SECONDS: f64 = 900.0;
const DEFAULT_TRAIN_STEP_CAP: usize = 1_000_000;
const DEFAULT_TRAIN_LOG_INTERVAL: usize = 500;

pub struct TrainConfig {
    pub seed: u64,
    pub step_cap: usize,
    pub log_interval: usize,
    pub eval_interval: Option<usize>,
    pub max_seconds: f64,
}

impl TrainConfig {
    pub fn from_env() -> Self {
        Self {
            seed: env_u64("TRAIN_SEED").unwrap_or(DEFAULT_SEED),
            step_cap: env_usize("TRAIN_STEPS").unwrap_or(DEFAULT_TRAIN_STEP_CAP),
            log_interval: env_usize("TRAIN_LOG_INTERVAL")
                .unwrap_or(DEFAULT_TRAIN_LOG_INTERVAL)
                .max(1),
            eval_interval: env_usize("TRAIN_EVAL_INTERVAL").filter(|interval| *interval > 0),
            max_seconds: env_f64("TRAIN_MAX_SECONDS")
                .filter(|seconds| *seconds > 0.0)
                .unwrap_or(DEFAULT_TRAIN_MAX_SECONDS),
        }
    }
}

pub fn load_model_path() -> Option<PathBuf> {
    env_nonempty("TRAIN_LOAD_MODEL").map(PathBuf::from)
}

pub fn save_model_path(run_output: &RunOutput) -> Option<PathBuf> {
    let value = env_nonempty("TRAIN_SAVE_MODEL")?;
    if value == "1" || value.eq_ignore_ascii_case("true") {
        Some(run_output.path("model.ckpt"))
    } else {
        Some(PathBuf::from(value))
    }
}

pub fn loss_graph_path(run_output: &RunOutput) -> PathBuf {
    env_nonempty("TRAIN_LOSS_GRAPH")
        .map(PathBuf::from)
        .unwrap_or_else(|| run_output.path("loss.png"))
}

pub fn generate_prompt() -> Option<String> {
    env_nonempty("TRAIN_GENERATE_PROMPT")
}

pub fn generate_tokens() -> usize {
    env_usize("TRAIN_GENERATE_TOKENS").unwrap_or(128)
}

pub fn sampling_config() -> SamplingConfig {
    SamplingConfig {
        temperature: env_f32("TRAIN_GENERATE_TEMPERATURE").unwrap_or(0.7),
        top_k: env_usize("TRAIN_GENERATE_TOP_K").unwrap_or(32),
        top_p: env_f32("TRAIN_GENERATE_TOP_P").unwrap_or(0.9),
    }
}

pub fn should_log_step(step: usize, step_cap: usize, log_interval: usize) -> bool {
    step == 0 || step + 1 == step_cap || step % log_interval == 0
}

pub fn should_eval_step(step: usize, step_cap: usize, eval_interval: Option<usize>) -> bool {
    eval_interval.is_some_and(|interval| step == 0 || step + 1 == step_cap || step % interval == 0)
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
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
