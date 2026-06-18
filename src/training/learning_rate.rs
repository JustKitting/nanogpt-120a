const TRAIN_LR_SCALE_ENV: &str = "TRAIN_LR_SCALE";
const TRAIN_ADAM_LR_SCALE_ENV: &str = "TRAIN_ADAM_LR_SCALE";
const TRAIN_LR_WARMUP_STEPS_ENV: &str = "TRAIN_LR_WARMUP_STEPS";
const TRAIN_LR_START_RATIO_ENV: &str = "TRAIN_LR_START_RATIO";
const TRAIN_AMUSE_BETA1_ENV: &str = "TRAIN_AMUSE_BETA1";
const TRAIN_AMUSE_RHO_ENV: &str = "TRAIN_AMUSE_RHO";
const DEFAULT_LR_WARMUP_STEPS: u32 = 5;
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

    let current = average_coefficient_for_step(step);
    let warmup = average_coefficient_for_step(warmup);
    let numerator = current * (1.0 - warmup);
    let denominator = warmup * (1.0 - current);
    if numerator <= 0.0 || denominator <= 0.0 {
        return beta1;
    }

    1.0 - (numerator / denominator).powf(amuse_rho()) * (1.0 - beta1)
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

fn average_coefficient_for_step(step: u32) -> f32 {
    let step = step.max(1);
    let weight = schedule_free_average_weight(step);
    let sum = schedule_free_weight_sum(step);
    if sum > 0.0 { weight / sum } else { 1.0 }
}

fn schedule_free_weight_sum(step: u32) -> f32 {
    let step = step.max(1);
    let warmup = warmup_steps();
    let warmup_terms = step.min(warmup);
    let post_warmup_terms = step.saturating_sub(warmup);
    warmup_weight_sum(warmup_terms, warmup, start_ratio()) + post_warmup_terms as f32
}

fn warmup_weight_sum(terms: u32, warmup: u32, start: f32) -> f32 {
    let n = terms as f64;
    let a = start as f64;
    let b = (1.0 - start as f64) / warmup as f64;
    let sum_i = n * (n + 1.0) * 0.5;
    let sum_i2 = n * (n + 1.0) * (2.0 * n + 1.0) / 6.0;
    (n * a * a + 2.0 * a * b * sum_i + b * b * sum_i2) as f32
}

fn warmup_steps() -> u32 {
    std::env::var(TRAIN_LR_WARMUP_STEPS_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_LR_WARMUP_STEPS)
        .max(1)
}

fn start_ratio() -> f32 {
    scale_from(TRAIN_LR_START_RATIO_ENV)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0)
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

fn scale_from(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}
