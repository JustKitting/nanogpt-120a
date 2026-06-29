use std::collections::HashSet;

use super::super::{candidate::Candidate, candidate_space, config::SweepConfig, rng::SweepRng};

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
        log_range(candidate.warmup_steps as f64, (5.0, 100.0)),
        range(
            candidate.start_ratio,
            candidate_space::START_RATIO_RANGE.0,
            candidate_space::START_RATIO_RANGE.1,
        ),
        range(
            candidate.amuse_beta1,
            candidate_space::AMUSE_BETA1_RANGE.0,
            candidate_space::AMUSE_BETA1_RANGE.1,
        ),
        range(
            candidate.amuse_rho,
            candidate_space::AMUSE_RHO_RANGE.0,
            candidate_space::AMUSE_RHO_RANGE.1,
        ),
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

#[cfg(test)]
mod tests {
    use super::{coverage_score, features};
    use crate::sweep::candidate::Candidate;

    #[test]
    fn coverage_score_prefers_uncovered_region() {
        let observed = [features(&candidate(4, 4, 0.5, 0.5, 5, 0.0, 0.2, 0.5))];
        let near = candidate(4, 4, 0.55, 0.55, 8, 0.02, 0.22, 0.55);
        let far = candidate(16, 8, 2.5, 2.5, 100, 0.2, 0.6, 1.0);

        assert!(coverage_score(&far, &observed) > coverage_score(&near, &observed));
    }

    fn candidate(
        batch_size: usize,
        n_layer: usize,
        lr_scale: f64,
        adam_lr_scale: f64,
        warmup_steps: usize,
        start_ratio: f64,
        amuse_beta1: f64,
        amuse_rho: f64,
    ) -> Candidate {
        Candidate {
            batch_size,
            n_layer,
            n_embd: if n_layer > 4 { 2048 } else { 1024 },
            n_head: 16,
            aurora_phases: if n_layer > 4 { 16 } else { 2 },
            aurora_blocks: if batch_size > 4 { 180 } else { 80 },
            lr_scale,
            adam_lr_scale,
            nextlat_lr_scale: 1.0,
            warmup_steps,
            start_ratio,
            amuse_beta1,
            amuse_rho,
        }
    }
}
