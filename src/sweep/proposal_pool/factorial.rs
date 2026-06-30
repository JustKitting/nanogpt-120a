use super::super::{
    analysis::{self, SweepAnalysis},
    candidate::Candidate,
    candidate_space as space,
    config::SweepConfig,
    rng::SweepRng,
};
use std::collections::HashSet;

#[cfg(test)]
mod tests;

pub fn candidates(
    used: &HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
    center: Option<&Candidate>,
    count: usize,
) -> Vec<Candidate> {
    let Some(center) = center else {
        return Vec::new();
    };
    let factors = uncertain_factors(config, analysis);
    if factors.is_empty() || count == 0 {
        return Vec::new();
    }
    let mut seen = used.clone();
    let mut out = Vec::new();
    let mut row = 0;
    while out.len() < count && row < count * 8 + 32 {
        let mut candidate = center.clone();
        for (index, factor) in factors.iter().enumerate() {
            set_factor(&mut candidate, factor, high_level(row, index));
        }
        fix_phase(&mut candidate, rng);
        if seen.insert(candidate.key()) {
            out.push(candidate);
        }
        row += 1;
    }
    out
}
fn uncertain_factors(config: &SweepConfig, analysis: &SweepAnalysis) -> Vec<String> {
    let mut factors = analysis::factor_beliefs(analysis, config)
        .into_iter()
        .map(|belief| {
            let score = belief.variance * (1.0 - belief.confidence).max(0.0);
            (belief.factor, score)
        })
        .filter(|(_, score)| *score > 0.0)
        .collect::<Vec<_>>();
    factors.sort_by(|a, b| b.1.total_cmp(&a.1));
    factors.into_iter().map(|(name, _)| name).collect()
}
fn set_factor(candidate: &mut Candidate, name: &str, high: bool) {
    match name {
        "batch_size" => candidate.batch_size = level(&space::BATCH_SIZE, high),
        "n_layer" => candidate.n_layer = level(&space::N_LAYER, high),
        "n_embd" => {
            let (n_embd, n_head) = level(&space::N_EMBD, high);
            candidate.n_embd = n_embd;
            candidate.n_head = n_head;
        }
        "aurora_blocks" => candidate.aurora_blocks = level(&space::AURORA_BLOCKS, high),
        "aurora_phases" => set_phase(candidate, high),
        "ln_lr_scale" => candidate.lr_scale = log_level(space::LR_SCALE_RANGE, high),
        "ln_adam_lr_scale" => {
            candidate.adam_lr_scale = log_level(space::LR_SCALE_RANGE, high);
        }
        "ln_nextlat_lr_scale" => {
            candidate.nextlat_lr_scale = log_level(space::LR_SCALE_RANGE, high);
        }
        "ln_warmup_steps" => {
            candidate.warmup_steps = usize_level(space::WARMUP_STEPS_RANGE, high);
        }
        "start_ratio" => {
            candidate.start_ratio = f64_level(space::START_RATIO_RANGE, high);
        }
        "amuse_beta1" => {
            candidate.amuse_beta1 = f64_level(space::AMUSE_BETA1_RANGE, high);
        }
        "amuse_rho" => candidate.amuse_rho = f64_level(space::AMUSE_RHO_RANGE, high),
        _ => {}
    }
}
fn high_level(row: usize, factor: usize) -> bool {
    ((row + 1) & (factor + 1)).count_ones() & 1 == 0
}
fn level<T: Copy>(values: &[T], high: bool) -> T {
    space::choose_unit(values, endpoint(high))
}
fn log_level(range: (f64, f64), high: bool) -> f64 {
    space::log_lerp(range, unit(high))
}
fn f64_level(range: (f64, f64), high: bool) -> f64 {
    space::range_f64(range, unit(high))
}
fn usize_level(range: (usize, usize), high: bool) -> usize {
    space::range_usize(range, unit(high))
}
fn unit(high: bool) -> f64 {
    if high { 0.875 } else { 0.125 }
}
fn endpoint(high: bool) -> f64 {
    if high { 1.0 } else { 0.0 }
}
fn set_phase(candidate: &mut Candidate, high: bool) {
    let phases = space::valid_aurora_phases(candidate.n_layer * 4, candidate.aurora_blocks);
    candidate.aurora_phases = if phases.is_empty() {
        candidate.aurora_phases
    } else if high {
        *phases.last().unwrap()
    } else {
        phases[0]
    };
}
fn fix_phase(candidate: &mut Candidate, rng: &mut SweepRng) {
    let phases = space::valid_aurora_phases(candidate.n_layer * 4, candidate.aurora_blocks);
    if !phases.contains(&candidate.aurora_phases) && !phases.is_empty() {
        candidate.aurora_phases = rng.choose(&phases);
    }
}
