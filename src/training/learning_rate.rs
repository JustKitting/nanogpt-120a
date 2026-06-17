const TRAIN_LR_SCALE_ENV: &str = "TRAIN_LR_SCALE";
const TRAIN_ADAM_LR_SCALE_ENV: &str = "TRAIN_ADAM_LR_SCALE";
const TRAIN_AURORA_LR_SCALE_ENV: &str = "TRAIN_AURORA_LR_SCALE";
const TRAIN_LR_WARMUP_STEPS_ENV: &str = "TRAIN_LR_WARMUP_STEPS";
const TRAIN_LR_STABLE_STEPS_ENV: &str = "TRAIN_LR_STABLE_STEPS";
const TRAIN_LR_DECAY_STEPS_ENV: &str = "TRAIN_LR_DECAY_STEPS";
const TRAIN_MIN_LR_RATIO_ENV: &str = "TRAIN_MIN_LR_RATIO";
const TRAIN_STEPS_ENV: &str = "TRAIN_STEPS";

pub(super) fn scale() -> f32 {
    scale_from(TRAIN_LR_SCALE_ENV).unwrap_or(1.0)
}

pub(super) fn adam_scale() -> f32 {
    scale_from(TRAIN_ADAM_LR_SCALE_ENV).unwrap_or_else(scale)
}

pub(super) fn adam_multiplier(step: u32) -> f32 {
    adam_scale() * warmup_stable_cosine(step)
}

pub(super) fn aurora_scale() -> f32 {
    scale_from(TRAIN_AURORA_LR_SCALE_ENV).unwrap_or_else(scale)
}

fn warmup_stable_cosine(step: u32) -> f32 {
    let step = step.max(1);
    let warmup = warmup_steps();
    let stable = stable_steps().max(warmup);
    let decay = decay_steps().max(stable + 1);
    let min_ratio = min_lr_ratio();

    if step <= warmup {
        return step as f32 / warmup as f32;
    }
    if step <= stable {
        return 1.0;
    }
    if step >= decay {
        return min_ratio;
    }

    let progress = (step - stable) as f32 / (decay - stable) as f32;
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    min_ratio + (1.0 - min_ratio) * cosine
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

fn stable_steps() -> u32 {
    std::env::var(TRAIN_LR_STABLE_STEPS_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(warmup_steps)
}

fn decay_steps() -> u32 {
    std::env::var(TRAIN_LR_DECAY_STEPS_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(train_steps)
}

fn min_lr_ratio() -> f32 {
    scale_from(TRAIN_MIN_LR_RATIO_ENV)
        .unwrap_or(0.1)
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
