use std::collections::HashSet;

use super::{PooledCandidate, coverage, direction, factorial, guided, local, variance};
use crate::sweep::{
    analysis::SweepAnalysis, candidate::Candidate, config::SweepConfig, rng::SweepRng,
};

pub(super) fn push_guided(
    pool: &mut Vec<PooledCandidate>,
    used: &mut HashSet<String>,
    rng: &mut SweepRng,
    direction: &direction::Direction,
    count: usize,
) {
    let target = pool.len() + count;
    let mut attempts = 0;
    while pool.len() < target && attempts < count.saturating_mul(32).max(32) {
        let jitter = attempts > 0;
        push_unique(
            pool,
            used,
            guided::candidate(rng, direction, jitter),
            "guided",
        );
        attempts += 1;
    }
}

pub(super) fn push_local(
    pool: &mut Vec<PooledCandidate>,
    used: &mut HashSet<String>,
    rng: &mut SweepRng,
    center: Option<&Candidate>,
    count: usize,
) {
    let target = (pool.len() + count).min(pool.capacity().max(1));
    push_candidates(pool, used, local::candidates(used, rng, center, target - pool.len()), "local");
}

pub(super) fn push_variance(
    pool: &mut Vec<PooledCandidate>,
    used: &mut HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
    count: usize,
) {
    let target = (pool.len() + count).min(config.candidate_samples.max(1));
    push_candidates(pool, used, variance::candidates(used, rng, config, analysis, target - pool.len()), "variance");
}

pub(super) fn push_coverage(
    pool: &mut Vec<PooledCandidate>,
    used: &mut HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    observed: &[Candidate],
    count: usize,
) {
    let target = (pool.len() + count).min(config.candidate_samples.max(1));
    push_candidates(pool, used, coverage::candidates(used, rng, config, observed, target - pool.len()), "coverage");
}

pub(super) fn push_factorial(
    pool: &mut Vec<PooledCandidate>,
    used: &mut HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
    center: Option<&Candidate>,
    count: usize,
) {
    let target = (pool.len() + count).min(config.candidate_samples.max(1));
    push_candidates(pool, used, factorial::candidates(used, rng, config, analysis, center, target - pool.len()), "factorial");
}

pub(super) fn push_random(
    pool: &mut Vec<PooledCandidate>,
    used: &mut HashSet<String>,
    rng: &mut SweepRng,
    target: usize,
) {
    while pool.len() < target {
        push_unique(pool, used, Candidate::random(rng), "random");
    }
}

fn push_unique(
    pool: &mut Vec<PooledCandidate>,
    used: &mut HashSet<String>,
    candidate: Candidate,
    source: &'static str,
) {
    if used.insert(candidate.key()) {
        pool.push(PooledCandidate { candidate, source });
    }
}

fn push_candidates(pool: &mut Vec<PooledCandidate>, used: &mut HashSet<String>, candidates: impl IntoIterator<Item = Candidate>, source: &'static str) {
    for candidate in candidates {
        push_unique(pool, used, candidate, source);
    }
}
