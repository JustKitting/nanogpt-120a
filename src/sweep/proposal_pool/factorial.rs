use super::super::{
    analysis::{self, SweepAnalysis},
    candidate::Candidate,
    candidate_space,
    config::SweepConfig,
    rng::SweepRng,
};
use std::collections::HashSet;
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
        "batch_size" => candidate.batch_size = level(&candidate_space::BATCH_SIZE, high),
        "n_layer" => candidate.n_layer = level(&candidate_space::N_LAYER, high),
        "n_embd" => {
            let (n_embd, n_head) = level(&candidate_space::N_EMBD, high);
            candidate.n_embd = n_embd;
            candidate.n_head = n_head;
        }
        "aurora_blocks" => {
            candidate.aurora_blocks = level(&candidate_space::AURORA_BLOCKS, high);
        }
        "aurora_phases" => set_phase(candidate, high),
        "ln_lr_scale" => {
            candidate.lr_scale = range_f64(candidate_space::LR_SCALE_RANGE, high, true);
        }
        "ln_adam_lr_scale" => {
            candidate.adam_lr_scale = range_f64(candidate_space::LR_SCALE_RANGE, high, true);
        }
        "ln_nextlat_lr_scale" => {
            candidate.nextlat_lr_scale = range_f64(candidate_space::LR_SCALE_RANGE, high, true);
        }
        "ln_warmup_steps" => {
            candidate.warmup_steps = range_usize(candidate_space::WARMUP_STEPS_RANGE, high);
        }
        "start_ratio" => {
            candidate.start_ratio = range_f64(candidate_space::START_RATIO_RANGE, high, false);
        }
        "amuse_beta1" => {
            candidate.amuse_beta1 = range_f64(candidate_space::AMUSE_BETA1_RANGE, high, false);
        }
        "amuse_rho" => {
            candidate.amuse_rho = range_f64(candidate_space::AMUSE_RHO_RANGE, high, false);
        }
        _ => {}
    }
}
fn level<T: Copy>(values: &[T], high: bool) -> T {
    values[if high { values.len() - 1 } else { 0 }]
}
fn high_level(row: usize, factor: usize) -> bool {
    ((row + 1) & (factor + 1)).count_ones() & 1 == 0
}
fn set_phase(candidate: &mut Candidate, high: bool) {
    let phases =
        candidate_space::valid_aurora_phases(candidate.n_layer * 4, candidate.aurora_blocks);
    candidate.aurora_phases = if phases.is_empty() {
        candidate.aurora_phases
    } else if high {
        *phases.last().unwrap()
    } else {
        phases[0]
    };
}
fn fix_phase(candidate: &mut Candidate, rng: &mut SweepRng) {
    let phases =
        candidate_space::valid_aurora_phases(candidate.n_layer * 4, candidate.aurora_blocks);
    if !phases.contains(&candidate.aurora_phases) && !phases.is_empty() {
        candidate.aurora_phases = rng.choose(&phases);
    }
}
fn range_f64(range: (f64, f64), high: bool, log_scale: bool) -> f64 {
    let t = if high { 0.875 } else { 0.125 };
    if log_scale {
        return (range.0.ln() + (range.1.ln() - range.0.ln()) * t).exp();
    }
    range.0 + (range.1 - range.0) * t
}
fn range_usize(range: (usize, usize), high: bool) -> usize {
    let t = if high { 0.875 } else { 0.125 };
    range.0 + (((range.1 - range.0) as f64) * t).round() as usize
}

#[cfg(test)]
mod tests {
    use super::high_level;

    #[test]
    fn hadamard_levels_are_balanced_per_factor() {
        let rows = 16;
        for factor in 0..8 {
            let highs = (0..rows).filter(|row| high_level(*row, factor)).count();
            assert_eq!(highs, rows / 2);
        }
    }

    #[test]
    fn hadamard_factor_pairs_cover_all_two_level_cells() {
        let rows = 16;
        for left in 0..4 {
            for right in left + 1..4 {
                let mut cells = [0; 4];
                for row in 0..rows {
                    let a = usize::from(high_level(row, left));
                    let b = usize::from(high_level(row, right));
                    cells[a * 2 + b] += 1;
                }
                assert_eq!(cells, [rows / 4; 4]);
            }
        }
    }
}
