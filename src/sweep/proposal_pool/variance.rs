use std::collections::HashSet;

use super::super::{
    analysis::{self, SweepAnalysis},
    candidate::Candidate,
    candidate_space,
    config::SweepConfig,
    rng::SweepRng,
};

mod halton;
#[cfg(test)]
mod tests;

pub fn candidates(
    used: &HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
    count: usize,
) -> Vec<Candidate> {
    let mut seen = used.clone();
    let mut ranked = Vec::new();
    let search = (config.candidate_samples.max(1) * 16).max(count * 8);
    let offset = rng.usize(8192);
    push_structured(&mut ranked, &mut seen, offset, search, config, analysis);
    push_random(&mut ranked, &mut seen, rng, count, config, analysis);
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked
        .into_iter()
        .take(count)
        .map(|(candidate, _)| candidate)
        .collect()
}

fn push_structured(
    ranked: &mut Vec<(Candidate, f64)>,
    seen: &mut HashSet<String>,
    offset: usize,
    search: usize,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
) {
    for index in 0..search {
        let candidate = candidate_space::from_unit(halton::units(offset + index + 1));
        if seen.insert(candidate.key()) {
            ranked.push(scored(candidate, config, analysis));
        }
    }
}

fn push_random(
    ranked: &mut Vec<(Candidate, f64)>,
    seen: &mut HashSet<String>,
    rng: &mut SweepRng,
    count: usize,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
) {
    while ranked.len() < count {
        let candidate = super::unique_random(seen, rng).unwrap_or_else(|| Candidate::random(rng));
        ranked.push(scored(candidate, config, analysis));
    }
}

fn scored(
    candidate: Candidate,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
) -> (Candidate, f64) {
    let score = analysis::score_candidate(analysis, config, &candidate);
    (candidate, score.uncertainty)
}
