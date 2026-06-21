mod direction;
mod guided;

#[cfg(test)]
mod tests;

use std::collections::HashSet;

use super::{analysis::SweepAnalysis, candidate::Candidate, config::SweepConfig, rng::SweepRng};

pub fn sample(
    seen: &HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
) -> Vec<Candidate> {
    let target = config.candidate_samples.max(1);
    let mut pool = Vec::with_capacity(target);
    let mut used = seen.clone();
    let direction = direction::from_analysis(analysis, config);
    push_guided(&mut pool, &mut used, rng, &direction, target.div_ceil(2));
    push_random(&mut pool, &mut used, rng, target);
    pool
}

fn push_guided(
    pool: &mut Vec<Candidate>,
    used: &mut HashSet<String>,
    rng: &mut SweepRng,
    direction: &direction::Direction,
    target: usize,
) {
    while pool.len() < target {
        let jitter = pool.len() > 0;
        push_unique(pool, used, guided::candidate(rng, direction, jitter));
    }
}

fn push_random(
    pool: &mut Vec<Candidate>,
    used: &mut HashSet<String>,
    rng: &mut SweepRng,
    target: usize,
) {
    while pool.len() < target {
        push_unique(pool, used, Candidate::random(rng));
    }
}

fn push_unique(pool: &mut Vec<Candidate>, used: &mut HashSet<String>, candidate: Candidate) {
    if used.insert(candidate.key()) {
        pool.push(candidate);
    }
}
