use std::collections::HashSet;

use super::super::{candidate::Candidate, candidate_space, config::SweepConfig, rng::SweepRng};

#[cfg(test)]
mod tests;

const FEATURE_COUNT: usize = 12;

pub fn candidates(
    used: &HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    observed: &[Candidate],
    count: usize,
) -> Vec<Candidate> {
    let mut seen = used.clone();
    let mut anchors = observed.iter().map(features).collect::<Vec<_>>();
    let mut out = Vec::new();
    let search = (config.candidate_samples.max(1) * 8).max(32);

    while out.len() < count {
        let Some(candidate) = best_candidate(&mut seen, rng, search, &anchors) else {
            break;
        };
        anchors.push(features(&candidate));
        out.push(candidate);
    }
    out
}

fn best_candidate(
    seen: &mut HashSet<String>,
    rng: &mut SweepRng,
    search: usize,
    anchors: &[[f64; FEATURE_COUNT]],
) -> Option<Candidate> {
    let mut best = None;
    let mut best_score = f64::NEG_INFINITY;
    for _ in 0..search {
        let candidate = unique_random(seen, rng)?;
        let score = coverage_score(&candidate, anchors);
        if score > best_score {
            best_score = score;
            best = Some(candidate);
        }
    }
    best
}

fn unique_random(seen: &mut HashSet<String>, rng: &mut SweepRng) -> Option<Candidate> {
    for _ in 0..4096 {
        let candidate = Candidate::random(rng);
        if seen.insert(candidate.key()) {
            return Some(candidate);
        }
    }
    None
}

fn coverage_score(candidate: &Candidate, anchors: &[[f64; FEATURE_COUNT]]) -> f64 {
    let x = features(candidate);
    if anchors.is_empty() {
        return distance2(&x, &[0.5; FEATURE_COUNT]);
    }
    anchors
        .iter()
        .map(|anchor| distance2(&x, anchor))
        .fold(f64::INFINITY, f64::min)
}

fn distance2(left: &[f64; FEATURE_COUNT], right: &[f64; FEATURE_COUNT]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = left - right;
            delta * delta
        })
        .sum()
}

fn features(candidate: &Candidate) -> [f64; FEATURE_COUNT] {
    [
        range(candidate.batch_size as f64, 4.0, 32.0),
        range(candidate.n_layer as f64, 4.0, 8.0),
        range(candidate.n_embd as f64, 1024.0, 2048.0),
        range(candidate.aurora_phases as f64, 2.0, 16.0),
        range(candidate.aurora_blocks as f64, 80.0, 180.0),
        log_range(candidate.lr_scale, candidate_space::LR_SCALE_RANGE),
        log_range(candidate.adam_lr_scale, candidate_space::LR_SCALE_RANGE),
        log_range(candidate.nextlat_lr_scale, candidate_space::LR_SCALE_RANGE),
        range_usize(candidate.warmup_steps, candidate_space::WARMUP_STEPS_RANGE),
        range_bounds(candidate.start_ratio, candidate_space::START_RATIO_RANGE),
        range_bounds(candidate.amuse_beta1, candidate_space::AMUSE_BETA1_RANGE),
        range_bounds(candidate.amuse_rho, candidate_space::AMUSE_RHO_RANGE),
    ]
}

fn range(value: f64, min: f64, max: f64) -> f64 {
    if max <= min {
        return 0.0;
    }
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

fn log_range(value: f64, bounds: (f64, f64)) -> f64 {
    range(value.ln(), bounds.0.ln(), bounds.1.ln())
}

fn range_bounds(value: f64, bounds: (f64, f64)) -> f64 {
    range(value, bounds.0, bounds.1)
}

fn range_usize(value: usize, bounds: (usize, usize)) -> f64 {
    range(value as f64, bounds.0 as f64, bounds.1 as f64)
}
