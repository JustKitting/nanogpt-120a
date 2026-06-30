use std::collections::HashSet;

use super::{
    analysis::{self, CandidateScore, SweepAnalysis},
    candidate::{Candidate, MIN_N_LAYER},
    config::SweepConfig,
    history::Trial,
    proposal_pool,
    rng::SweepRng,
};

const NAN_PENALTY_LOSS: f64 = 1.0e6;
const FAILED_TRIAL_PENALTY_LOSS: f64 = 5.0e5;

#[cfg(test)]
mod tests;

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

fn select_candidate<'a>(
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

fn best_local_center(trials: &[Trial], config: &SweepConfig) -> Option<Candidate> {
    best_screen_candidate(trials, config).or_else(|| best_full_candidate(trials, config))
}

fn best_screen_candidate(trials: &[Trial], config: &SweepConfig) -> Option<Candidate> {
    trials
        .iter()
        .filter_map(|trial| {
            let loss = trial.screen_val_loss?;
            if !loss.is_finite() || trial.candidate.n_layer < MIN_N_LAYER {
                return None;
            }
            if !time_budget_matches(trial.screen_elapsed_s, config.screen_max_seconds) {
                return None;
            }
            Some((loss, trial.candidate.with_min_layers()))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, candidate)| candidate)
}

fn best_full_candidate(trials: &[Trial], config: &SweepConfig) -> Option<Candidate> {
    trials
        .iter()
        .filter_map(|trial| {
            let loss = trial.val_loss?;
            if !loss.is_finite() || trial.candidate.n_layer < MIN_N_LAYER {
                return None;
            }
            if !time_budget_matches(trial.elapsed_s, config.max_seconds) {
                return None;
            }
            Some((loss, trial.candidate.with_min_layers()))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, candidate)| candidate)
}

fn time_budget_matches(elapsed_s: Option<f64>, target_s: f64) -> bool {
    let Some(elapsed_s) = elapsed_s else {
        return false;
    };
    elapsed_s.is_finite() && elapsed_s >= target_s * 0.8 && elapsed_s <= target_s * 1.25
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

fn observed_loss(trial: &Trial) -> Option<f64> {
    if trial.candidate.n_layer < MIN_N_LAYER {
        return None;
    }
    if trial.status == "dry_run" {
        return None;
    }
    if trial.status == "failed_build" || trial.status == "failed_run" {
        return Some(FAILED_TRIAL_PENALTY_LOSS);
    }
    if trial.status == "rejected_screen" {
        return trial.screen_val_loss.or(Some(FAILED_TRIAL_PENALTY_LOSS));
    }
    if trial.status.starts_with("nan") {
        return Some(NAN_PENALTY_LOSS);
    }
    trial.val_loss
}

fn infeasible_build_shapes(trials: &[Trial], config: &SweepConfig) -> HashSet<String> {
    trials
        .iter()
        .filter(|trial| trial.status == "failed_build" || trial.status == "failed_run")
        .filter(|trial| {
            let elapsed = trial.screen_elapsed_s.or(trial.elapsed_s).unwrap_or(0.0);
            elapsed == 0.0 || elapsed >= config.screen_max_seconds * 0.95
        })
        .map(|trial| trial.candidate.build_key())
        .collect()
}

fn unseen_random(
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
