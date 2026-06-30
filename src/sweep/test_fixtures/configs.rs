use std::path::PathBuf;

use crate::sweep::config::SweepConfig;

pub(in crate::sweep) fn config(random_trials: usize, candidate_samples: usize) -> SweepConfig {
    SweepConfig {
        trials: 4,
        random_trials,
        candidate_samples,
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

pub(in crate::sweep) fn quality_config(candidate_samples: usize) -> SweepConfig {
    SweepConfig {
        sweep_stability_weight: 0.0,
        sweep_exploration_weight: 0.0,
        ..config(0, candidate_samples)
    }
}
