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

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, path::PathBuf};

    use super::{best_local_center, infeasible_build_shapes, unseen_random};
    use crate::sweep::{candidate::Candidate, config::SweepConfig, history::Trial, rng::SweepRng};

    #[test]
    fn marks_build_shape_infeasible_after_failed_run() {
        let config = config();
        let candidate = candidate(32, 8, 2048, 16, 180, 1.0);
        let shapes = infeasible_build_shapes(
            &[Trial {
                candidate: candidate.clone(),
                status: "failed_run".to_string(),
                val_loss: None,
                completed_steps: None,
                elapsed_s: Some(0.0),
                screen_val_loss: None,
                screen_completed_steps: None,
                screen_elapsed_s: None,
                screen_reason: None,
                log_path: PathBuf::from("screen.log"),
            }],
            &config,
        );

        assert!(shapes.contains(&candidate.build_key()));
    }

    #[test]
    fn random_candidate_skips_known_infeasible_build_shape() {
        let mut rng = SweepRng::new(0x4750_5432);
        let mut infeasible = HashSet::new();
        let bad = candidate(32, 8, 2048, 16, 180, 1.0);
        infeasible.insert(bad.build_key());

        for _ in 0..64 {
            let candidate = unseen_random(&HashSet::new(), &mut rng, &infeasible);
            assert!(!infeasible.contains(&candidate.build_key()));
        }
    }

    #[test]
    fn local_center_uses_best_timed_screen_result() {
        let config = config();
        let best = screen_trial(candidate(16, 4, 1024, 8, 180, 2.309_529), 6.340_408);
        let b32 = screen_trial(candidate(32, 4, 2048, 16, 180, 2.013_4), 7.034_256);
        let stale_longer_b32 = Trial {
            screen_elapsed_s: Some(180.0),
            screen_val_loss: Some(5.129_354),
            ..screen_trial(candidate(32, 4, 1024, 16, 180, 1.984_246), 5.129_354)
        };
        let incomplete = Trial {
            screen_elapsed_s: Some(8.0),
            screen_val_loss: Some(5.0),
            ..screen_trial(candidate(8, 4, 1024, 8, 120, 1.5), 5.0)
        };

        let center =
            best_local_center(&[stale_longer_b32, b32, incomplete, best.clone()], &config).unwrap();

        assert_eq!(center.batch_size, best.candidate.batch_size);
        assert_eq!(center.n_layer, best.candidate.n_layer);
        assert_eq!(center.n_embd, best.candidate.n_embd);
        assert_eq!(center.lr_scale, best.candidate.lr_scale);
    }

    fn candidate(
        batch_size: usize,
        n_layer: usize,
        n_embd: usize,
        aurora_phases: usize,
        aurora_blocks: usize,
        lr_scale: f64,
    ) -> Candidate {
        Candidate {
            batch_size,
            n_layer,
            n_embd,
            n_head: 16,
            aurora_phases,
            aurora_blocks,
            lr_scale,
            adam_lr_scale: 1.0,
            nextlat_lr_scale: 1.0,
            warmup_steps: 20,
            start_ratio: 0.1,
            amuse_beta1: 0.4,
            amuse_rho: 0.8,
        }
    }

    fn screen_trial(candidate: Candidate, screen_loss: f64) -> Trial {
        Trial {
            candidate,
            status: "rejected_screen".to_string(),
            val_loss: None,
            completed_steps: None,
            elapsed_s: None,
            screen_val_loss: Some(screen_loss),
            screen_completed_steps: Some(100),
            screen_elapsed_s: Some(30.0),
            screen_reason: Some("screen_loss_worse".to_string()),
            log_path: PathBuf::from("screen.log"),
        }
    }

    fn config() -> SweepConfig {
        SweepConfig {
            trials: 4,
            random_trials: 0,
            candidate_samples: 16,
            max_seconds: 900.0,
            screen_max_seconds: 30.0,
            sweep_quality_weight: 1.0,
            sweep_stability_weight: 0.75,
            sweep_exploration_weight: 0.35,
            log_interval: 500,
            dataset: "synth".to_string(),
            arch: "sm_120a".to_string(),
            cuda_device: None,
            sweep_dir: None,
            seed_history: PathBuf::from("notes/sweep_seed_current.tsv"),
            baseline: PathBuf::from("notes/sweep_baseline.env"),
            seed: 0x4750_5432,
            dry_run: false,
        }
    }
}
