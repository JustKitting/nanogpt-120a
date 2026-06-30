use super::super::{
    analysis::{self, SweepAnalysis},
    candidate::Candidate,
    config::SweepConfig,
    rng::SweepRng,
};
use std::collections::HashSet;

mod factor;
mod pattern;
#[cfg(test)]
mod tests;

pub fn candidates(
    used: &HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
    center: Option<&Candidate>,
    count: usize,
) -> Vec<Candidate> {
    let Some(center) = center else {
        return Vec::new();
    };
    let factors = uncertain_factors(config, analysis);
    if factors.is_empty() || count == 0 {
        return Vec::new();
    }
    let mut seen = used.clone();
    let mut out = Vec::new();
    let mut row = 0;
    while out.len() < count && row < count * 8 + 32 {
        let mut candidate = center.clone();
        for (index, factor) in factors.iter().enumerate() {
            factor::set(&mut candidate, factor, pattern::high_level(row, index));
        }
        factor::fix_phase(&mut candidate, rng);
        if seen.insert(candidate.key()) {
            out.push(candidate);
        }
        row += 1;
    }
    out
}
fn uncertain_factors(config: &SweepConfig, analysis: &SweepAnalysis) -> Vec<String> {
    let mut factors = analysis::factor_beliefs(analysis, config)
        .into_iter()
        .map(|belief| {
            let score = belief.variance * (1.0 - belief.confidence).max(0.0);
            (belief.factor, score)
        })
        .filter(|(_, score)| *score > 0.0)
        .collect::<Vec<_>>();
    factors.sort_by(|a, b| b.1.total_cmp(&a.1));
    factors.into_iter().map(|(name, _)| name).collect()
}
