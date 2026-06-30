use std::collections::HashSet;

use crate::sweep::{
    analysis, candidate::Candidate, candidate_space, config::SweepConfig, history::Trial,
    rng::SweepRng,
};

use super::halton;

#[test]
fn halton_units_cover_each_factor_range() {
    let rows = (1..=128).map(halton::units).collect::<Vec<_>>();
    for dim in 0..candidate_space::FACTOR_COUNT {
        let min = rows
            .iter()
            .map(|row| row[dim])
            .fold(f64::INFINITY, f64::min);
        let max = rows
            .iter()
            .map(|row| row[dim])
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(min < 0.15, "dim={dim} min={min}");
        assert!(max > 0.85, "dim={dim} max={max}");
    }
}

#[test]
fn variance_candidates_are_unique_structured_points() {
    let config = config();
    let trials = [
        trial(candidate(4, 4), 5.0),
        trial(candidate(4, 8), 4.0),
        trial(candidate(16, 4), 4.0),
        trial(candidate(16, 8), 3.0),
    ];
    let analysis = analysis::analyze(&trials, &config);
    let candidates = super::candidates(
        &HashSet::new(),
        &mut SweepRng::new(0x9911),
        &config,
        &analysis,
        8,
    );
    let unique = candidates
        .iter()
        .map(|candidate| candidate.key())
        .collect::<HashSet<_>>();

    assert_eq!(candidates.len(), 8);
    assert_eq!(unique.len(), candidates.len());
}

fn trial(candidate: Candidate, val_loss: f64) -> Trial {
    Trial {
        candidate,
        status: "success".to_string(),
        val_loss: Some(val_loss),
        completed_steps: Some(10),
        elapsed_s: Some(900.0),
        screen_val_loss: Some(val_loss + 1.0),
        screen_completed_steps: Some(10),
        screen_elapsed_s: Some(30.0),
        screen_reason: Some("screen_loss_improved".to_string()),
        log_path: "train.log".into(),
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
        nextlat_lr_scale: 1.0,
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
        candidate_samples: 24,
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
        seed_history: "notes/sweep_seed_current.tsv".into(),
        baseline: "notes/sweep_baseline.env".into(),
        seed: 0x4750_5432,
        dry_run: false,
    }
}
