mod budget;
mod coverage;
mod direction;
mod factorial;
mod fill;
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
    debug_assert_eq!(budget.total(), target);
    fill::push_guided(&mut pool, &mut used, rng, &direction, budget.guided);
    fill::push_local(&mut pool, &mut used, rng, center, budget.local);
    fill::push_factorial(
        &mut pool,
        &mut used,
        rng,
        config,
        analysis,
        center,
        budget.factorial,
    );
    fill::push_variance(&mut pool, &mut used, rng, config, analysis, budget.variance);
    fill::push_coverage(&mut pool, &mut used, rng, config, observed, budget.coverage);
    fill::push_random(&mut pool, &mut used, rng, target);
    pool
}
