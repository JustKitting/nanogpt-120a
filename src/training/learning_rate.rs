const TRAIN_LR_SCALE_ENV: &str = "TRAIN_LR_SCALE";
const TRAIN_ADAM_LR_SCALE_ENV: &str = "TRAIN_ADAM_LR_SCALE";
const TRAIN_LR_WARMUP_STEPS_ENV: &str = "TRAIN_LR_WARMUP_STEPS";
const TRAIN_AMUSE_BETA1_ENV: &str = "TRAIN_AMUSE_BETA1";
const TRAIN_AMUSE_RHO_ENV: &str = "TRAIN_AMUSE_RHO";
const TRAIN_STEPS_ENV: &str = "TRAIN_STEPS";
const AMUSE_AVERAGE_R: f32 = 0.0;
const AMUSE_WEIGHT_LR_POWER: f32 = 2.0;

pub(super) fn scale() -> f32 {
    scale_from(TRAIN_LR_SCALE_ENV).unwrap_or(1.0)
}

pub(super) fn adam_scale() -> f32 {
    scale_from(TRAIN_ADAM_LR_SCALE_ENV).unwrap_or_else(scale)
}

pub(super) fn adam_multiplier(step: u32) -> f32 {
    adam_scale() * warmup_only(step)
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
        step as f32 / warmup as f32
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
        .unwrap_or_else(default_warmup_steps)
        .max(1)
}

fn default_warmup_steps() -> u32 {
    train_steps().div_ceil(20).clamp(1, 2_000)
}

fn amuse_beta1() -> f32 {
    scale_from(TRAIN_AMUSE_BETA1_ENV)
        .unwrap_or(0.4)
        .clamp(0.0, 1.0)
}

fn amuse_rho() -> f32 {
    scale_from(TRAIN_AMUSE_RHO_ENV)
        .unwrap_or(0.8)
        .clamp(0.0, 1.0)
}

fn train_steps() -> u32 {
    std::env::var(TRAIN_STEPS_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10)
}

fn scale_from(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}
