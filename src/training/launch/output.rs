use std::fs;
use std::path::{Path, PathBuf};

use gpt2_nvfp4::{
    GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_N_LAYER, GPT2_SEQ_LEN, GPT2_TOKEN_ROWS,
};
use rust_kernels_cuda::optimizer::{AURORA_COOPERATIVE_BLOCKS, AURORA_MATRIX_PHASES};
use time::OffsetDateTime;

use super::{TrainConfig, env_nonempty};
use crate::AppResult;

const RUNS_DIR: &str = "target/runs";

#[derive(Clone)]
pub(super) struct RunOutput {
    dir: PathBuf,
}

impl RunOutput {
    pub(super) fn new(dataset: &str, label: &str) -> AppResult<Self> {
        let dir = default_run_dir(dataset, label);
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    pub(super) fn dir(&self) -> &Path {
        &self.dir
    }

    pub(super) fn path(&self, file_name: &str) -> PathBuf {
        self.dir.join(file_name)
    }

    pub(super) fn write_info(&self, info: &str) -> AppResult {
        fs::write(self.path("run_info.txt"), info)?;
        Ok(())
    }
}

pub(super) fn save_model_path(run_output: &RunOutput) -> Option<PathBuf> {
    let value = env_nonempty("TRAIN_SAVE_MODEL")?;
    if value == "1" || value.eq_ignore_ascii_case("true") {
        Some(run_output.path("model.ckpt"))
    } else {
        Some(PathBuf::from(value))
    }
}

pub(super) fn write_generated_text(run_output: &RunOutput, text: &str) -> AppResult<PathBuf> {
    let path = run_output.path("generated.txt");
    ensure_parent(&path)?;
    fs::write(&path, text)?;
    Ok(path)
}

pub(super) fn build_run_info(dataset: &str, config: &TrainConfig) -> String {
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

pub(super) fn ensure_parent(path: &Path) -> AppResult {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
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
