use std::{fs, path::PathBuf, str::FromStr, sync::OnceLock};

use crate::env_file;

const TRAIN_LR_SCALE_ENV: &str = "TRAIN_LR_SCALE";
const TRAIN_ADAM_LR_SCALE_ENV: &str = "TRAIN_ADAM_LR_SCALE";
const TRAIN_NEXTLAT_LR_SCALE_ENV: &str = "TRAIN_NEXTLAT_LR_SCALE";
const TRAIN_LR_WARMUP_STEPS_ENV: &str = "TRAIN_LR_WARMUP_STEPS";
const TRAIN_LR_START_RATIO_ENV: &str = "TRAIN_LR_START_RATIO";
const TRAIN_AMUSE_BETA1_ENV: &str = "TRAIN_AMUSE_BETA1";
const TRAIN_AMUSE_RHO_ENV: &str = "TRAIN_AMUSE_RHO";

const DEFAULT_LR_SCALE: f32 = 1.014_04;
const DEFAULT_ADAM_LR_SCALE: f32 = 1.980_467;
const DEFAULT_NEXTLAT_LR_SCALE: f32 = 1.0;
const DEFAULT_LR_WARMUP_STEPS: u32 = 5;
const DEFAULT_LR_START_RATIO: f32 = 0.05;
const DEFAULT_AMUSE_BETA1: f32 = 0.2;
const DEFAULT_AMUSE_RHO: f32 = 0.5;

pub(in crate::training) fn scale() -> f32 {
    config_value(TRAIN_LR_SCALE_ENV, DEFAULT_LR_SCALE)
}

pub(in crate::training) fn adam_scale() -> f32 {
    config_value(TRAIN_ADAM_LR_SCALE_ENV, DEFAULT_ADAM_LR_SCALE)
}

pub(in crate::training) fn next_latent_scale() -> f32 {
    config_value(TRAIN_NEXTLAT_LR_SCALE_ENV, DEFAULT_NEXTLAT_LR_SCALE)
}

pub(in crate::training) fn warmup_steps() -> u32 {
    config_value(TRAIN_LR_WARMUP_STEPS_ENV, DEFAULT_LR_WARMUP_STEPS).max(1)
}

pub(in crate::training) fn start_ratio() -> f32 {
    config_value(TRAIN_LR_START_RATIO_ENV, DEFAULT_LR_START_RATIO).clamp(0.0, 1.0)
}

pub(in crate::training) fn amuse_beta1() -> f32 {
    config_value(TRAIN_AMUSE_BETA1_ENV, DEFAULT_AMUSE_BETA1).clamp(0.0, 1.0)
}

pub(in crate::training) fn amuse_rho() -> f32 {
    config_value(TRAIN_AMUSE_RHO_ENV, DEFAULT_AMUSE_RHO).clamp(0.0, 1.0)
}

fn config_value<T: FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .or_else(|| env_file::parsed(baseline_text(), name))
        .unwrap_or(default)
}

fn baseline_text() -> &'static str {
    static BASELINE: OnceLock<String> = OnceLock::new();
    BASELINE
        .get_or_init(|| fs::read_to_string(baseline_path()).unwrap_or_default())
        .as_str()
}

fn baseline_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("notes/sweep_baseline.env")
}
