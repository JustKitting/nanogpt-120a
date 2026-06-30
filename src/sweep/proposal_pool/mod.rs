mod budget;
mod coverage;
mod direction;
mod factorial;
mod guided;
mod local;
mod variance;

#[cfg(test)]
mod tests;

use std::collections::HashSet;

use super::{analysis::SweepAnalysis, candidate::Candidate, config::SweepConfig, rng::SweepRng};
use budget::source_budget;

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
    observed: &[Candidate],
) -> Vec<PooledCandidate> {
    let target = config.candidate_samples.max(1);
    let mut pool = Vec::with_capacity(target);
    let mut used = seen.clone();
    let direction = direction::from_analysis(analysis, config);
    let budget = source_budget(target, analysis, config);
    debug_assert_eq!(
        budget.guided
            + budget.local
            + budget.factorial
            + budget.variance
            + budget.coverage
            + budget.random,
        target
    );
    push_guided(&mut pool, &mut used, rng, &direction, budget.guided);
    push_local(&mut pool, &mut used, rng, center, budget.local);
    push_factorial(
        &mut pool,
        &mut used,
        rng,
        config,
        analysis,
        center,
        budget.factorial,
    );
    push_variance(&mut pool, &mut used, rng, config, analysis, budget.variance);
    push_coverage(&mut pool, &mut used, rng, config, observed, budget.coverage);
    push_random(&mut pool, &mut used, rng, target);
    pool
}

fn push_guided(
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

fn push_local(
    pool: &mut Vec<PooledCandidate>,
    used: &mut HashSet<String>,
    rng: &mut SweepRng,
    center: Option<&Candidate>,
    count: usize,
) {
    let target = (pool.len() + count).min(pool.capacity().max(1));
    for candidate in local::candidates(used, rng, center, target - pool.len()) {
        push_unique(pool, used, candidate, "local");
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

fn push_coverage(
    pool: &mut Vec<PooledCandidate>,
    used: &mut HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    observed: &[Candidate],
    count: usize,
) {
    let target = (pool.len() + count).min(config.candidate_samples.max(1));
    for candidate in coverage::candidates(used, rng, config, observed, target - pool.len()) {
        push_unique(pool, used, candidate, "coverage");
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
