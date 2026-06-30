use std::collections::HashSet;

use crate::sweep::{candidate::Candidate, rng::SweepRng};

use super::ScoredCandidate;

pub(super) fn select_candidate<'a>(
    ranked: &'a [ScoredCandidate],
    rng: &mut SweepRng,
) -> Option<&'a Candidate> {
    let source = select_source(ranked, rng)?;
    ranked
        .iter()
        .find(|scored| scored.source == source)
        .or_else(|| ranked.first())
        .map(|scored| &scored.candidate)
}

fn select_source(ranked: &[ScoredCandidate], rng: &mut SweepRng) -> Option<&'static str> {
    let sources = [
        "guided",
        "local",
        "factorial",
        "variance",
        "coverage",
        "random",
    ];
    let counts = sources.map(|source| {
        ranked
            .iter()
            .filter(|candidate| candidate.source == source)
            .count()
    });
    let total = counts.iter().sum::<usize>();
    if total == 0 {
        return None;
    }

    let mut ticket = rng.usize(total);
    for (source, count) in sources.into_iter().zip(counts) {
        if ticket < count {
            return Some(source);
        }
        ticket -= count;
    }
    None
}

pub(super) fn unseen_random(
    seen: &HashSet<String>,
    rng: &mut SweepRng,
    infeasible_builds: &HashSet<String>,
) -> Candidate {
    for _ in 0..4096 {
        let candidate = Candidate::random(rng);
        if !seen.contains(&candidate.key()) && !infeasible_builds.contains(&candidate.build_key()) {
            return candidate;
        }
    }
    Candidate::random(rng)
}
