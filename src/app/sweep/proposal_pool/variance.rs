use std::collections::HashSet;

use super::super::{
    analysis::{self, SweepAnalysis},
    candidate::Candidate,
    config::SweepConfig,
    rng::SweepRng,
};

pub fn candidates(
    used: &HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
    count: usize,
) -> Vec<Candidate> {
    let mut seen = used.clone();
    let mut ranked = Vec::new();
    let search = (config.candidate_samples.max(1) * 4).max(count);
    for _ in 0..search {
        let candidate = unique_random(&mut seen, rng);
        let score = analysis::score_candidate(analysis, config, &candidate);
        ranked.push((candidate, score.uncertainty));
    }
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked
        .into_iter()
        .take(count)
        .map(|(candidate, _)| candidate)
        .collect()
}

fn unique_random(seen: &mut HashSet<String>, rng: &mut SweepRng) -> Candidate {
    for _ in 0..4096 {
        let candidate = Candidate::random(rng);
        if seen.insert(candidate.key()) {
            return candidate;
        }
    }
    Candidate::random(rng)
}
