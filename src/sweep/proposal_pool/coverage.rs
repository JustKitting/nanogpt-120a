use std::collections::HashSet;

use super::super::features::{FEATURE_COUNT, unit_features};
use super::super::{candidate::Candidate, config::SweepConfig, rng::SweepRng};

#[cfg(test)]
mod tests;

pub fn candidates(
    used: &HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    observed: &[Candidate],
    count: usize,
) -> Vec<Candidate> {
    let mut seen = used.clone();
    let mut anchors = observed.iter().map(unit_features).collect::<Vec<_>>();
    let mut out = Vec::new();
    let search = (config.candidate_samples.max(1) * 8).max(32);

    while out.len() < count {
        let Some(candidate) = best_candidate(&mut seen, rng, search, &anchors) else {
            break;
        };
        anchors.push(unit_features(&candidate));
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
        let candidate = super::unique_random(seen, rng)?;
        let score = coverage_score(&candidate, anchors);
        if score > best_score {
            best_score = score;
            best = Some(candidate);
        }
    }
    best
}

fn coverage_score(candidate: &Candidate, anchors: &[[f64; FEATURE_COUNT]]) -> f64 {
    let x = unit_features(candidate);
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
