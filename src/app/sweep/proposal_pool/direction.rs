use super::super::{analysis::SweepAnalysis, config::SweepConfig};

#[derive(Default)]
pub struct Direction {
    pub batch_size: f64,
    pub n_layer: f64,
    pub n_embd: f64,
    pub aurora_phases: f64,
    pub aurora_blocks: f64,
    pub lr_scale: f64,
    pub adam_lr_scale: f64,
    pub warmup_steps: f64,
    pub start_ratio: f64,
    pub amuse_beta1: f64,
    pub amuse_rho: f64,
}

pub fn from_analysis(analysis: &SweepAnalysis, config: &SweepConfig) -> Direction {
    let mut direction = Direction::default();
    for response in &analysis.models {
        let weight = response_weight(response.name, config);
        for effect in response
            .model
            .effects
            .iter()
            .filter(|effect| !effect.name.contains('*'))
        {
            add(&mut direction, &effect.name, effect.coefficient * weight);
        }
    }
    direction
}

fn response_weight(name: &str, config: &SweepConfig) -> f64 {
    if name.contains("quality") {
        config.sweep_quality_weight
    } else if name.contains("speed") {
        config.sweep_speed_weight
    } else if name == "stability" {
        config.sweep_stability_weight
    } else {
        0.0
    }
}

fn add(direction: &mut Direction, name: &str, value: f64) {
    match name {
        "batch_size" => direction.batch_size += value,
        "n_layer" => direction.n_layer += value,
        "n_embd" => direction.n_embd += value,
        "aurora_phases" => direction.aurora_phases += value,
        "aurora_blocks" => direction.aurora_blocks += value,
        "ln_lr_scale" => direction.lr_scale += value,
        "ln_adam_lr_scale" => direction.adam_lr_scale += value,
        "ln_warmup_steps" => direction.warmup_steps += value,
        "start_ratio" => direction.start_ratio += value,
        "amuse_beta1" => direction.amuse_beta1 += value,
        "amuse_rho" => direction.amuse_rho += value,
        _ => {}
    }
}
