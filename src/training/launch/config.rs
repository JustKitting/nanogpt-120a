use std::path::PathBuf;

use super::super::SamplingConfig;

const DEFAULT_SEED: u64 = 0x4750_5432;
const DEFAULT_TRAIN_MAX_SECONDS: f64 = 900.0;
const DEFAULT_TRAIN_STEP_CAP: usize = 1_000_000;
const AUTO_GENERATE_MIN_TRAIN_SECONDS: f64 = 900.0;
const DEFAULT_SYNTH_PROMPT: &str = "The";
const DEFAULT_SHAKESPEARE_PROMPT: &str = "KING:";

#[derive(Clone, Copy)]
pub(in crate::training) struct TrainConfig {
    pub(in crate::training) seed: u64,
    pub(in crate::training) step_cap: usize,
    pub(in crate::training) log_interval: usize,
    pub(in crate::training) eval_interval: Option<usize>,
    pub(in crate::training) max_seconds: f64,
}

impl TrainConfig {
    pub(super) fn from_env() -> Self {
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

pub(super) fn load_model_path() -> Option<PathBuf> {
    env_nonempty("TRAIN_LOAD_MODEL").map(PathBuf::from)
}

pub(super) fn generate_prompt(dataset: &str, train_elapsed_s: f64) -> Option<String> {
    env_nonempty("TRAIN_GENERATE_PROMPT").or_else(|| {
        (train_elapsed_s >= AUTO_GENERATE_MIN_TRAIN_SECONDS)
            .then(|| default_generate_prompt(dataset).to_string())
    })
}

pub(super) fn generate_tokens() -> usize {
    env_usize("TRAIN_GENERATE_TOKENS").unwrap_or(128)
}

pub(super) fn sampling_config() -> SamplingConfig {
    SamplingConfig {
        temperature: env_f32("TRAIN_GENERATE_TEMPERATURE").unwrap_or(0.7),
        top_k: env_usize("TRAIN_GENERATE_TOP_K").unwrap_or(32),
        top_p: env_f32("TRAIN_GENERATE_TOP_P").unwrap_or(0.9),
    }
}

pub(super) fn should_log_step(step: usize, step_cap: usize, log_interval: usize) -> bool {
    step == 0 || step + 1 == step_cap || step % log_interval == 0
}

pub(super) fn should_eval_step(step: usize, step_cap: usize, eval_interval: Option<usize>) -> bool {
    eval_interval.is_some_and(|interval| step == 0 || step + 1 == step_cap || step % interval == 0)
}

pub(in crate::training) fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

pub(in crate::training) fn env_bool(name: &str) -> Option<bool> {
    let value = std::env::var(name).ok()?;
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
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
