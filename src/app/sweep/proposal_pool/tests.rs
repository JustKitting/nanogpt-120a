use std::{collections::HashSet, path::PathBuf};

use super::super::{analysis, candidate::Candidate, config::SweepConfig, history::Trial};

#[test]
fn guided_pool_uses_main_effect_direction() {
    let config = config();
    let trials = [
        trial(candidate(4, 4), 5.0),
        trial(candidate(4, 8), 3.0),
        trial(candidate(16, 4), 3.0),
        trial(candidate(16, 8), 1.0),
    ];
    let analysis = analysis::analyze(&trials, &config);
    let center = candidate(8, 4);
    let pool = super::sample(
        &HashSet::new(),
        &mut super::super::rng::SweepRng::new(0x1234),
        &config,
        &analysis,
        Some(&center),
    );

    assert_eq!(pool[0].source, "guided");
    assert_eq!(pool[0].candidate.batch_size, 16);
    assert_eq!(pool[0].candidate.n_layer, 8);
    assert!(pool.iter().any(|candidate| candidate.source == "factorial"));
    assert!(pool.iter().any(|candidate| candidate.source == "variance"));
    assert!(pool.iter().any(|candidate| candidate.source == "random"));
}

fn trial(candidate: Candidate, val_loss: f64) -> Trial {
    Trial {
        candidate,
        status: "success".to_string(),
        val_loss: Some(val_loss),
        completed_steps: Some(10),
        log_path: PathBuf::from("train.log"),
    }
}

fn candidate(batch_size: usize, n_layer: usize) -> Candidate {
    Candidate {
        batch_size,
        n_layer,
        n_embd: 1024,
        n_head: 16,
        aurora_phases: 4,
        aurora_blocks: 80,
        lr_scale: 1.0,
        adam_lr_scale: 1.0,
        warmup_steps: 20,
        start_ratio: 0.1,
        amuse_beta1: 0.4,
        amuse_rho: 0.8,
    }
}

fn config() -> SweepConfig {
    SweepConfig {
        trials: 4,
        random_trials: 0,
        candidate_samples: 8,
        max_seconds: 900.0,
        screen_steps: 500,
        screen_max_seconds: 180.0,
        sweep_quality_weight: 1.0,
        sweep_speed_weight: 0.0,
        sweep_stability_weight: 0.0,
        sweep_exploration_weight: 0.0,
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
