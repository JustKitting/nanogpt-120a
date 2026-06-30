use std::path::PathBuf;

use super::{candidate::Candidate, config::SweepConfig, history::Trial, rng::SweepRng};

pub(in crate::sweep) fn trial(status: &str, val_loss: Option<f64>, candidate: Candidate) -> Trial {
    Trial {
        candidate,
        status: status.to_string(),
        val_loss,
        completed_steps: Some(10),
        elapsed_s: Some(900.0),
        screen_val_loss: val_loss.map(|loss| loss + 1.0),
        screen_completed_steps: Some(10),
        screen_elapsed_s: Some(30.0),
        screen_reason: Some("screen_loss_improved".to_string()),
        log_path: PathBuf::from("train.log"),
    }
}

pub(in crate::sweep) fn success_trial(candidate: Candidate, val_loss: f64) -> Trial {
    trial("success", Some(val_loss), candidate)
}

pub(in crate::sweep) fn trial_with_losses(
    candidate: Candidate,
    val_loss: f64,
    screen_loss: f64,
) -> Trial {
    Trial {
        screen_val_loss: Some(screen_loss),
        ..success_trial(candidate, val_loss)
    }
}

pub(in crate::sweep) fn trial_with_status(candidate: Candidate, status: &str) -> Trial {
    Trial {
        screen_val_loss: None,
        screen_completed_steps: None,
        screen_elapsed_s: None,
        screen_reason: None,
        ..trial(status, None, candidate)
    }
}

pub(in crate::sweep) fn candidate(batch_size: usize, n_layer: usize, lr_scale: f64) -> Candidate {
    Candidate {
        batch_size,
        n_layer,
        n_embd: 1536,
        n_head: 12,
        aurora_phases: 8,
        aurora_blocks: 180,
        lr_scale,
        adam_lr_scale: 1.0,
        nextlat_lr_scale: 1.0,
        warmup_steps: 5,
        start_ratio: 0.0,
        amuse_beta1: 0.4,
        amuse_rho: 0.8,
    }
}

pub(in crate::sweep) fn basic_candidate(batch_size: usize, n_layer: usize) -> Candidate {
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

pub(in crate::sweep) fn measured_candidate() -> Candidate {
    Candidate {
        batch_size: 8,
        n_layer: 2,
        n_embd: 1024,
        n_head: 16,
        aurora_phases: 2,
        aurora_blocks: 80,
        lr_scale: 1.014_040,
        adam_lr_scale: 1.980_467,
        nextlat_lr_scale: 1.0,
        warmup_steps: 5,
        start_ratio: 0.05,
        amuse_beta1: 0.2,
        amuse_rho: 0.5,
    }
}

pub(in crate::sweep) fn temp_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("{}-{}-{name}", std::process::id(), nanos()));
    path
}

pub(in crate::sweep) fn rng() -> SweepRng {
    SweepRng::new(0x4750_5432)
}

fn nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}
