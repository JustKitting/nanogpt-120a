use std::collections::HashSet;

use super::{
    candidate::{Candidate, MIN_N_LAYER},
    history::Trial,
    rng::SweepRng,
};

const NAN_PENALTY_LOSS: f64 = 1.0e6;
const SCREEN_REJECT_PENALTY_LOSS: f64 = 1.0e5;

pub fn propose(
    trials: &[Trial],
    seen: &HashSet<String>,
    rng: &mut SweepRng,
    random_trials: usize,
    samples: usize,
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
    if completed < random_trials {
        return unseen_random(seen, rng);
    }

    let mut ranked = trials
        .iter()
        .filter_map(|trial| Some((score_loss(trial)?, encode(&trial.candidate))))
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    let split = ranked.len().div_ceil(4).max(1);
    let good = ranked[..split]
        .iter()
        .map(|(_, x)| x.clone())
        .collect::<Vec<_>>();
    let bad = ranked[split..]
        .iter()
        .map(|(_, x)| x.clone())
        .collect::<Vec<_>>();

    let mut best = unseen_random(seen, rng);
    let mut best_score = f64::NEG_INFINITY;
    for _ in 0..samples.max(1) {
        let candidate = unseen_random(seen, rng);
        let x = encode(&candidate);
        let score = kde_log_density(&x, &good) - kde_log_density(&x, &bad);
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

fn encode(candidate: &Candidate) -> Vec<f64> {
    vec![
        candidate.batch_size as f64,
        candidate.n_layer as f64,
        candidate.n_embd as f64 / 1024.0,
        candidate.n_head as f64,
        candidate.aurora_phases as f64,
        candidate.aurora_blocks as f64 / 120.0,
        candidate.lr_scale.ln(),
        candidate.adam_lr_scale.ln(),
        candidate.warmup_steps as f64 / 20.0,
        candidate.start_ratio,
        candidate.amuse_beta1,
        candidate.amuse_rho,
    ]
}

fn kde_log_density(x: &[f64], points: &[Vec<f64>]) -> f64 {
    if points.is_empty() {
        return -64.0;
    }
    let mut total = 0.0;
    for point in points {
        let mut dist = 0.0;
        for (a, b) in x.iter().zip(point) {
            let d = (a - b) / 0.75;
            dist += d * d;
        }
        total += (-0.5 * dist).exp();
    }
    (total / points.len() as f64 + 1.0e-12).ln()
}
