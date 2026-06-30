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
