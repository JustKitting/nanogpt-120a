use std::collections::HashSet;

use super::{
    analysis::{self, CandidateScore, SweepAnalysis},
    candidate::Candidate,
    config::SweepConfig,
    history::Trial,
    proposal_pool,
    rng::SweepRng,
};

mod select;
#[cfg(test)]
mod tests;
mod trial;

use select::{select_candidate, unseen_random};
use trial::{best_local_center, infeasible_build_shapes, observed_loss};

#[derive(Clone, Debug)]
pub struct Proposal {
    pub candidate: Candidate,
    pub reason: &'static str,
    pub ranked: Vec<ScoredCandidate>,
}

#[derive(Clone, Debug)]
pub struct ScoredCandidate {
    pub candidate: Candidate,
    pub source: &'static str,
    pub score: CandidateScore,
}

pub fn propose(
    trials: &[Trial],
    seen: &HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
    baseline: Option<&Candidate>,
) -> Proposal {
    let infeasible_builds = infeasible_build_shapes(trials, config);
    if let Some(candidate) = baseline {
        let candidate = candidate.with_min_layers();
        if !seen.contains(&candidate.key()) && !infeasible_builds.contains(&candidate.build_key()) {
            return proposal("baseline", candidate, analysis, config);
        }
    }

    let completed = trials
        .iter()
        .filter(|trial| observed_loss(trial).is_some())
        .count();
    if completed < config.random_trials {
        return proposal(
            "random",
            unseen_random(seen, rng, &infeasible_builds),
            analysis,
            config,
        );
    }

    let center =
        best_local_center(trials, config).or_else(|| baseline.map(Candidate::with_min_layers));
    let observed = trials
        .iter()
        .map(|trial| trial.candidate.clone())
        .collect::<Vec<_>>();
    let mut ranked = proposal_pool::sample(seen, rng, config, analysis, center.as_ref(), &observed)
        .into_iter()
        .filter(|pooled| !infeasible_builds.contains(&pooled.candidate.build_key()))
        .map(|pooled| {
            let score = analysis::score_candidate(analysis, config, &pooled.candidate);
            ScoredCandidate {
                candidate: pooled.candidate,
                source: pooled.source,
                score,
            }
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.score.score.total_cmp(&a.score.score));
    let candidate = select_candidate(&ranked, rng)
        .cloned()
        .unwrap_or_else(|| unseen_random(seen, rng, &infeasible_builds));
    Proposal {
        candidate,
        reason: "model",
        ranked,
    }
}

fn proposal(
    reason: &'static str,
    candidate: Candidate,
    analysis: &SweepAnalysis,
    config: &SweepConfig,
) -> Proposal {
    let score = analysis::score_candidate(analysis, config, &candidate);
    Proposal {
        candidate: candidate.clone(),
        reason,
        ranked: vec![ScoredCandidate {
            candidate,
            source: reason,
            score,
        }],
    }
}
