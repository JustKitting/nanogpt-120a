use std::path::PathBuf;

use crate::sweep::candidate::Candidate;

use super::record::Record;

pub(super) fn record(text: &str) -> Option<Record> {
    Some(Record {
        val_loss: value(text, "VAL_LOSS")?.parse().ok()?,
        completed_steps: value(text, "COMPLETED_STEPS").and_then(|value| value.parse().ok()),
        elapsed_s: value(text, "TRAIN_ELAPSED_S").and_then(|value| value.parse().ok()),
        screen_loss: value(text, "SCREEN_LOSS").and_then(|value| value.parse().ok()),
        screen_completed_steps: value(text, "SCREEN_COMPLETED_STEPS")
            .and_then(|value| value.parse().ok()),
        screen_elapsed_s: value(text, "SCREEN_ELAPSED_S").and_then(|value| value.parse().ok()),
        screen_reason: value(text, "SCREEN_REASON").map(ToString::to_string),
        log_path: PathBuf::from(value(text, "LOG_PATH").unwrap_or("")),
        candidate: Candidate {
            batch_size: value(text, "GPT2_BATCH_SIZE")?.parse().ok()?,
            n_layer: value(text, "GPT2_N_LAYER")?.parse().ok()?,
            n_embd: value(text, "GPT2_N_EMBD")?.parse().ok()?,
            n_head: value(text, "GPT2_N_HEAD")?.parse().ok()?,
            aurora_phases: value(text, "AURORA_MATRIX_PHASES")?.parse().ok()?,
            aurora_blocks: value(text, "AURORA_COOPERATIVE_BLOCKS")?.parse().ok()?,
            lr_scale: value(text, "TRAIN_LR_SCALE")?.parse().ok()?,
            adam_lr_scale: value(text, "TRAIN_ADAM_LR_SCALE")?.parse().ok()?,
            nextlat_lr_scale: value(text, "TRAIN_NEXTLAT_LR_SCALE")
                .and_then(|value| value.parse().ok())
                .unwrap_or(1.0),
            warmup_steps: value(text, "TRAIN_LR_WARMUP_STEPS")?.parse().ok()?,
            start_ratio: value(text, "TRAIN_LR_START_RATIO")?.parse().ok()?,
            amuse_beta1: value(text, "TRAIN_AMUSE_BETA1")?.parse().ok()?,
            amuse_rho: value(text, "TRAIN_AMUSE_RHO")?.parse().ok()?,
        },
    })
}

fn value<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    text.lines().find_map(|line| {
        let (key, value) = line.split_once('=')?;
        (key.trim() == name).then_some(value.trim())
    })
}
