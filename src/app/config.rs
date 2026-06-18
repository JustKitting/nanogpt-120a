use std::path::PathBuf;

use crate::training::SamplingConfig;

use super::run_output::RunOutput;

pub const SEED: u64 = 0x4750_5432;
const DEFAULT_TRAIN_STEPS: usize = 10;

pub struct TrainConfig {
    pub steps: usize,
    pub log_interval: usize,
    pub eval_interval: Option<usize>,
}

impl TrainConfig {
    pub fn from_env() -> Self {
        Self {
            steps: env_usize("TRAIN_STEPS").unwrap_or(DEFAULT_TRAIN_STEPS),
            log_interval: env_usize("TRAIN_LOG_INTERVAL").unwrap_or(1).max(1),
            eval_interval: env_usize("TRAIN_EVAL_INTERVAL").filter(|interval| *interval > 0),
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

pub fn should_log_step(step: usize, steps: usize, log_interval: usize) -> bool {
    step == 0 || step + 1 == steps || step % log_interval == 0
}

pub fn should_eval_step(step: usize, steps: usize, eval_interval: Option<usize>) -> bool {
    eval_interval.is_some_and(|interval| step == 0 || step + 1 == steps || step % interval == 0)
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}
