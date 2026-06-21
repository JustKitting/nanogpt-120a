use std::collections::HashSet;

use super::{
    analysis::{self, CandidateScore, SweepAnalysis},
    candidate::{Candidate, MIN_N_LAYER},
    config::SweepConfig,
    history::Trial,
    rng::SweepRng,
};

const NAN_PENALTY_LOSS: f64 = 1.0e6;
const SCREEN_REJECT_PENALTY_LOSS: f64 = 1.0e5;

#[derive(Clone, Debug)]
pub struct Proposal {
    pub candidate: Candidate,
    pub reason: &'static str,
    pub ranked: Vec<ScoredCandidate>,
}

#[derive(Clone, Debug)]
pub struct ScoredCandidate {
    pub candidate: Candidate,
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
    if let Some(candidate) = baseline {
        let candidate = candidate.with_min_layers();
        if !seen.contains(&candidate.key()) {
            return proposal("baseline", candidate, analysis, config);
        }
    }

    let completed = trials
        .iter()
        .filter(|trial| score_loss(trial).is_some())
        .count();
    if completed < config.random_trials {
        return proposal("random", unseen_random(seen, rng), analysis, config);
    }

    let mut ranked = Vec::new();
    let mut sample_seen = seen.clone();
    for _ in 0..config.candidate_samples.max(1) {
        let candidate = unseen_random(&sample_seen, rng);
        sample_seen.insert(candidate.key());
        let score = analysis::score_candidate(analysis, config, &candidate);
        ranked.push(ScoredCandidate { candidate, score });
    }
    ranked.sort_by(|a, b| b.score.score.total_cmp(&a.score.score));
    let candidate = ranked
        .first()
        .map(|scored| scored.candidate.clone())
        .unwrap_or_else(|| unseen_random(seen, rng));
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
        ranked: vec![ScoredCandidate { candidate, score }],
    }
}

fn score_loss(trial: &Trial) -> Option<f64> {
    if trial.candidate.n_layer < MIN_N_LAYER {
        return None;
    }
    if trial.status == "dry_run" {
        return None;
    }
    if trial.status == "rejected_screen" {
        return Some(SCREEN_REJECT_PENALTY_LOSS);
    }
    if trial.status.starts_with("nan") {
        return Some(NAN_PENALTY_LOSS);
    }
    trial.val_loss
}

fn unseen_random(seen: &HashSet<String>, rng: &mut SweepRng) -> Candidate {
    for _ in 0..4096 {
        let candidate = Candidate::random(rng);
        if !seen.contains(&candidate.key()) {
            return candidate;
        }
    }
    Candidate::random(rng)
}
