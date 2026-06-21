use std::collections::HashSet;

use super::{
    analysis::{self, SweepAnalysis},
    candidate::{Candidate, MIN_N_LAYER},
    config::SweepConfig,
    history::Trial,
    rng::SweepRng,
};

const NAN_PENALTY_LOSS: f64 = 1.0e6;
const SCREEN_REJECT_PENALTY_LOSS: f64 = 1.0e5;

pub fn propose(
    trials: &[Trial],
    seen: &HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
    baseline: Option<&Candidate>,
) -> Candidate {
    if let Some(candidate) = baseline {
        let candidate = candidate.with_min_layers();
        if !seen.contains(&candidate.key()) {
            return candidate;
        }
    }

    let completed = trials
        .iter()
        .filter(|trial| score_loss(trial).is_some())
        .count();
    if completed < config.random_trials {
        return unseen_random(seen, rng);
    }

    let mut best = unseen_random(seen, rng);
    let mut best_score = f64::NEG_INFINITY;
    for _ in 0..config.candidate_samples.max(1) {
        let candidate = unseen_random(seen, rng);
        let score = analysis::score_candidate(analysis, config, &candidate).score;
        if score > best_score {
            best_score = score;
            best = candidate;
        }
    }
    best
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
