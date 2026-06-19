use crate::app::config::{SEED, TrainConfig};

pub fn build(dataset: &str, config: &TrainConfig) -> String {
    let mut info = String::new();
    push_info(&mut info, "dataset", dataset);
    push_info(&mut info, "tokenizer", llama2_tokenizer::TOKENIZER_NAME);
    push_info(&mut info, "vocab_size", llama2_tokenizer::VOCAB_SIZE);
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
    push_info(&mut info, "seed", format!("{SEED:#x}"));
    push_run_env(&mut info);
    info
}

fn push_run_env(info: &mut String) {
    for name in [
        "CUDA_DEVICE_INDEX",
        "TRAIN_LOAD_MODEL",
        "TRAIN_SAVE_MODEL",
        "TRAIN_MAX_SECONDS",
        "TRAIN_REPEAT_BATCH",
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
