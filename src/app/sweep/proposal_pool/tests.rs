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

#[test]
fn factorial_pool_can_probe_more_than_four_supported_factors() {
    let config = config();
    let trials = (0..24)
        .map(|i| trial(wide_candidate(i), 24.0 - i as f64))
        .collect::<Vec<_>>();
    let analysis = analysis::analyze(&trials, &config);
    let center = wide_candidate(0);
    let pool = super::sample(
        &HashSet::new(),
        &mut super::super::rng::SweepRng::new(0x8822),
        &config,
        &analysis,
        Some(&center),
    );
    let factorial = pool
        .iter()
        .find(|candidate| candidate.source == "factorial")
        .unwrap();

    assert!(changed_factors(&center, &factorial.candidate) > 4);
}

fn trial(candidate: Candidate, val_loss: f64) -> Trial {
    Trial {
        candidate,
        status: "success".to_string(),
        val_loss: Some(val_loss),
        completed_steps: Some(10),
        elapsed_s: Some(5.0),
        screen_val_loss: Some(val_loss + 1.0),
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

fn changed_factors(left: &Candidate, right: &Candidate) -> usize {
    usize::from(left.batch_size != right.batch_size)
        + usize::from(left.n_layer != right.n_layer)
        + usize::from(left.n_embd != right.n_embd)
        + usize::from(left.aurora_phases != right.aurora_phases)
        + usize::from(left.aurora_blocks != right.aurora_blocks)
        + usize::from(left.lr_scale != right.lr_scale)
        + usize::from(left.adam_lr_scale != right.adam_lr_scale)
        + usize::from(left.warmup_steps != right.warmup_steps)
        + usize::from(left.start_ratio != right.start_ratio)
        + usize::from(left.amuse_beta1 != right.amuse_beta1)
        + usize::from(left.amuse_rho != right.amuse_rho)
}

fn wide_candidate(i: usize) -> Candidate {
    Candidate {
        batch_size: [4, 8, 16][i % 3],
        n_layer: [4, 8][(i / 3) % 2],
        n_embd: [1024, 2048][(i / 5) % 2],
        n_head: 16,
        aurora_phases: [4, 8, 16][(i / 7) % 3],
        aurora_blocks: [80, 90, 120, 160, 180][(i / 11) % 5],
        lr_scale: 0.5 + (i % 11) as f64 * 0.18,
        adam_lr_scale: 0.5 + (i % 13) as f64 * 0.15,
        warmup_steps: 5 + (i * 7) % 96,
        start_ratio: (i % 9) as f64 * 0.025,
        amuse_beta1: 0.2 + (i % 7) as f64 * 0.06,
        amuse_rho: 0.5 + (i % 6) as f64 * 0.08,
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
