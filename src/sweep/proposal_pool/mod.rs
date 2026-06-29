mod coverage;
mod direction;
mod factorial;
mod guided;
mod local;
mod variance;

#[cfg(test)]
mod tests;

use std::collections::HashSet;

use super::{
    analysis::{self, SweepAnalysis},
    candidate::Candidate,
    config::SweepConfig,
    rng::SweepRng,
};

#[derive(Clone, Debug)]
pub struct PooledCandidate {
    pub candidate: Candidate,
    pub source: &'static str,
}

#[derive(Clone, Copy, Debug, Default)]
struct SourceBudget {
    guided: usize,
    local: usize,
    factorial: usize,
    variance: usize,
    coverage: usize,
    random: usize,
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

fn source_budget(target: usize, analysis: &SweepAnalysis, config: &SweepConfig) -> SourceBudget {
    let mut weights = source_weights(analysis, config);
    if target < 4 {
        weights.coverage += weights.random;
        weights.random = 0.0;
    }

    normalized_budget(target, weights)
}

#[derive(Clone, Copy, Debug)]
struct SourceWeights {
    guided: f64,
    local: f64,
    factorial: f64,
    variance: f64,
    coverage: f64,
    random: f64,
}

fn source_weights(analysis: &SweepAnalysis, config: &SweepConfig) -> SourceWeights {
    let beliefs = analysis::factor_beliefs(analysis, config);
    let confidence = if beliefs.is_empty() {
        0.0
    } else {
        beliefs.iter().map(|belief| belief.confidence).sum::<f64>() / beliefs.len() as f64
    }
    .clamp(0.0, 1.0);
    let variance = if beliefs.is_empty() {
        1.0
    } else {
        beliefs.iter().map(|belief| belief.variance).sum::<f64>() / beliefs.len() as f64
    }
    .clamp(0.0, 1.0);
    let model_maturity =
        (analysis.trial_count as f64 / (analysis.trial_count as f64 + 12.0)).clamp(0.0, 1.0);
    let has_response_model = analysis
        .models
        .iter()
        .any(|model| model.name.contains("quality"));
    let exploitation = if has_response_model {
        (0.25 + 0.55 * model_maturity * confidence).clamp(0.0, 0.8)
    } else {
        0.0
    };
    let uncertainty = (1.0 - confidence).max(variance.sqrt()).clamp(0.0, 1.0);

    SourceWeights {
        guided: exploitation,
        local: if has_response_model {
            (0.2 + 0.3 * model_maturity).clamp(0.0, 0.45)
        } else {
            0.0
        },
        factorial: 0.15 + 0.25 * uncertainty,
        variance: 0.2 + 0.35 * uncertainty,
        coverage: 0.15 + 0.3 * (1.0 - model_maturity),
        random: 0.1 + 0.2 * (1.0 - model_maturity),
    }
}

fn normalized_budget(target: usize, weights: SourceWeights) -> SourceBudget {
    let raw = [
        weights.guided,
        weights.local,
        weights.factorial,
        weights.variance,
        weights.coverage,
        weights.random,
    ];
    let total = raw.iter().sum::<f64>();
    if total <= 0.0 {
        return SourceBudget {
            random: target,
            ..SourceBudget::default()
        };
    }

    let mut counts = [0usize; 6];
    let mut remainders = [(0usize, 0.0); 6];
    for (index, weight) in raw.iter().enumerate() {
        if *weight <= 0.0 {
            continue;
        }
        let exact = target as f64 * *weight / total;
        counts[index] = exact.floor() as usize;
        if counts[index] == 0 {
            counts[index] = 1;
        }
        remainders[index] = (index, exact - exact.floor());
    }

    while counts.iter().sum::<usize>() > target {
        let Some(index) = counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .min_by(|a, b| remainders[a.0].1.total_cmp(&remainders[b.0].1))
            .map(|(index, _)| index)
        else {
            break;
        };
        counts[index] -= 1;
    }
    while counts.iter().sum::<usize>() < target {
        remainders.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (index, _) in remainders {
            if counts.iter().sum::<usize>() >= target {
                break;
            }
            if raw[index] > 0.0 {
                counts[index] += 1;
            }
        }
    }

    SourceBudget {
        guided: counts[0],
        local: counts[1],
        factorial: counts[2],
        variance: counts[3],
        coverage: counts[4],
        random: counts[5],
    }
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
