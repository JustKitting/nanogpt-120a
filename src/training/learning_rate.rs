use std::{fs, path::PathBuf, sync::OnceLock};

const TRAIN_LR_SCALE_ENV: &str = "TRAIN_LR_SCALE";
const TRAIN_ADAM_LR_SCALE_ENV: &str = "TRAIN_ADAM_LR_SCALE";
const TRAIN_NEXTLAT_LR_SCALE_ENV: &str = "TRAIN_NEXTLAT_LR_SCALE";
const TRAIN_LR_WARMUP_STEPS_ENV: &str = "TRAIN_LR_WARMUP_STEPS";
const TRAIN_LR_START_RATIO_ENV: &str = "TRAIN_LR_START_RATIO";
const TRAIN_AMUSE_BETA1_ENV: &str = "TRAIN_AMUSE_BETA1";
const TRAIN_AMUSE_RHO_ENV: &str = "TRAIN_AMUSE_RHO";
const DEFAULT_LR_SCALE: f32 = 1.014_040;
const DEFAULT_ADAM_LR_SCALE: f32 = 1.980_467;
const DEFAULT_NEXTLAT_LR_SCALE: f32 = 1.0;
const DEFAULT_LR_WARMUP_STEPS: u32 = 5;
const DEFAULT_LR_START_RATIO: f32 = 0.05;
const DEFAULT_AMUSE_BETA1: f32 = 0.2;
const DEFAULT_AMUSE_RHO: f32 = 0.5;
const AMUSE_AVERAGE_R: f32 = 0.0;
const AMUSE_WEIGHT_LR_POWER: f32 = 2.0;

pub(super) fn scale() -> f32 {
    scale_from(TRAIN_LR_SCALE_ENV)
        .or_else(|| baseline().f32(TRAIN_LR_SCALE_ENV))
        .unwrap_or(DEFAULT_LR_SCALE)
}

pub(super) fn adam_scale() -> f32 {
    scale_from(TRAIN_ADAM_LR_SCALE_ENV)
        .or_else(|| baseline().f32(TRAIN_ADAM_LR_SCALE_ENV))
        .unwrap_or(DEFAULT_ADAM_LR_SCALE)
}

pub(super) fn next_latent_scale() -> f32 {
    scale_from(TRAIN_NEXTLAT_LR_SCALE_ENV)
        .or_else(|| baseline().f32(TRAIN_NEXTLAT_LR_SCALE_ENV))
        .unwrap_or(DEFAULT_NEXTLAT_LR_SCALE)
}

pub(super) fn adam_multiplier(step: u32) -> f32 {
    adam_scale() * warmup_only(step)
}

pub(super) fn next_latent_adam_multiplier(step: u32) -> f32 {
    adam_multiplier(step) * next_latent_scale()
}

pub(super) fn aurora_multiplier(step: u32) -> f32 {
    scale() * warmup_only(step)
}

pub(super) fn schedule_free_beta(step: u32) -> f32 {
    let step = step.max(1);
    let warmup = warmup_steps().max(2);
    let beta1 = amuse_beta1();
    if step <= warmup {
        return beta1;
    }

    let ratio = (warmup - 1) as f32 / (step - 1) as f32;
    1.0 - ratio.powf(amuse_rho()) * (1.0 - beta1)
}

pub(super) fn schedule_free_average_coefficient(step: u32, weight_sum: &mut f32) -> f32 {
    let weight = schedule_free_average_weight(step);
    *weight_sum += weight;
    if *weight_sum > 0.0 {
        weight / *weight_sum
    } else {
        0.0
    }
}

fn warmup_only(step: u32) -> f32 {
    let step = step.max(1);
    let warmup = warmup_steps();
    if step <= warmup {
        let progress = step as f32 / warmup as f32;
        let start = start_ratio();
        start + (1.0 - start) * progress
    } else {
        1.0
    }
}

fn schedule_free_average_weight(step: u32) -> f32 {
    let step = step.max(1);
    let lr_max = warmup_only(step);
    (step as f32).powf(AMUSE_AVERAGE_R) * lr_max.powf(AMUSE_WEIGHT_LR_POWER)
}

fn warmup_steps() -> u32 {
    std::env::var(TRAIN_LR_WARMUP_STEPS_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .or_else(|| baseline().u32(TRAIN_LR_WARMUP_STEPS_ENV))
        .unwrap_or(DEFAULT_LR_WARMUP_STEPS)
        .max(1)
}

fn start_ratio() -> f32 {
    scale_from(TRAIN_LR_START_RATIO_ENV)
        .or_else(|| baseline().f32(TRAIN_LR_START_RATIO_ENV))
        .unwrap_or(DEFAULT_LR_START_RATIO)
        .clamp(0.0, 1.0)
}

fn amuse_beta1() -> f32 {
    scale_from(TRAIN_AMUSE_BETA1_ENV)
        .or_else(|| baseline().f32(TRAIN_AMUSE_BETA1_ENV))
        .unwrap_or(DEFAULT_AMUSE_BETA1)
        .clamp(0.0, 1.0)
}

fn amuse_rho() -> f32 {
    scale_from(TRAIN_AMUSE_RHO_ENV)
        .or_else(|| baseline().f32(TRAIN_AMUSE_RHO_ENV))
        .unwrap_or(DEFAULT_AMUSE_RHO)
        .clamp(0.0, 1.0)
}

fn scale_from(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

fn baseline() -> &'static Baseline {
    static BASELINE: OnceLock<Baseline> = OnceLock::new();
    BASELINE.get_or_init(Baseline::load)
}

struct Baseline {
    text: String,
}

impl Baseline {
    fn load() -> Self {
        Self {
            text: fs::read_to_string(baseline_path()).unwrap_or_default(),
        }
    }

    fn f32(&self, name: &str) -> Option<f32> {
        self.value(name).and_then(|value| value.parse().ok())
    }

    fn u32(&self, name: &str) -> Option<u32> {
        self.value(name).and_then(|value| value.parse().ok())
    }

    fn value(&self, name: &str) -> Option<&str> {
        self.text.lines().find_map(|line| {
            let (key, value) = line.split_once('=')?;
            (key.trim() == name).then_some(value.trim())
        })
    }
}

fn baseline_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("notes/sweep_baseline.env")
}
