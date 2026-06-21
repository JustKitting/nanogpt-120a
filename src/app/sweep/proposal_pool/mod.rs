mod direction;
mod factorial;
mod guided;
mod variance;

#[cfg(test)]
mod tests;

use std::collections::HashSet;

use super::{analysis::SweepAnalysis, candidate::Candidate, config::SweepConfig, rng::SweepRng};

#[derive(Clone, Debug)]
pub struct PooledCandidate {
    pub candidate: Candidate,
    pub source: &'static str,
}

pub fn sample(
    seen: &HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
    center: Option<&Candidate>,
) -> Vec<PooledCandidate> {
    let target = config.candidate_samples.max(1);
    let mut pool = Vec::with_capacity(target);
    let mut used = seen.clone();
    let direction = direction::from_analysis(analysis, config);
    push_guided(&mut pool, &mut used, rng, &direction, target.div_ceil(4));
    push_factorial(
        &mut pool,
        &mut used,
        rng,
        config,
        analysis,
        center,
        target.div_ceil(4),
    );
    push_variance(
        &mut pool,
        &mut used,
        rng,
        config,
        analysis,
        target.div_ceil(4),
    );
    push_random(&mut pool, &mut used, rng, target);
    pool
}

fn push_guided(
    pool: &mut Vec<PooledCandidate>,
    used: &mut HashSet<String>,
    rng: &mut SweepRng,
    direction: &direction::Direction,
    target: usize,
) {
    while pool.len() < target {
        let jitter = pool.len() > 0;
        push_unique(
            pool,
            used,
            guided::candidate(rng, direction, jitter),
            "guided",
        );
    }
}

fn push_variance(
    pool: &mut Vec<PooledCandidate>,
    used: &mut HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
    count: usize,
) {
    let target = (pool.len() + count).min(config.candidate_samples.max(1));
    for candidate in variance::candidates(used, rng, config, analysis, target - pool.len()) {
        push_unique(pool, used, candidate, "variance");
    }
}

fn push_factorial(
    pool: &mut Vec<PooledCandidate>,
    used: &mut HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
    center: Option<&Candidate>,
    count: usize,
) {
    let target = (pool.len() + count).min(config.candidate_samples.max(1));
    for candidate in factorial::candidates(used, rng, config, analysis, center, target - pool.len())
    {
        push_unique(pool, used, candidate, "factorial");
    }
}

fn push_random(
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
