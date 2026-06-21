use std::path::PathBuf;

use super::super::{candidate::Candidate, config::SweepConfig, history::Trial};

#[test]
fn scoring_uses_pairwise_interaction_signal() {
    let config = config();
    let trials = [
        trial(candidate(4, 4), 1.0),
        trial(candidate(16, 8), 1.0),
        trial(candidate(4, 8), 9.0),
        trial(candidate(16, 4), 9.0),
    ];
    let analysis = super::analyze(&trials, &config);
    let aligned = super::score_candidate(&analysis, &config, &candidate(16, 8));
    let crossed = super::score_candidate(&analysis, &config, &candidate(16, 4));

    let aligned_quality = aligned.predicted_quality.unwrap().standard_score;
    let crossed_quality = crossed.predicted_quality.unwrap().standard_score;
    assert!(aligned_quality > crossed_quality + 0.5);
    assert!(analysis.models.iter().any(|model| {
        model.name == "full_quality"
            && model
                .model
                .effects
                .iter()
                .any(|effect| effect.name == "batch_size*n_layer")
    }));
}

#[test]
fn scoring_reports_expected_improvement_against_best_observed_quality() {
    let config = config();
    let trials = [
        trial(candidate(4, 4), 9.0),
        trial(candidate(4, 8), 5.0),
        trial(candidate(16, 4), 5.0),
        trial(candidate(16, 8), 1.0),
    ];
    let analysis = super::analyze(&trials, &config);
    let best_like = super::score_candidate(&analysis, &config, &candidate(16, 8));
    let bad_like = super::score_candidate(&analysis, &config, &candidate(4, 4));

    assert!((0.0..=1.0).contains(&best_like.probability_improvement));
    assert!(best_like.expected_improvement.is_finite());
    assert!(best_like.probability_improvement > bad_like.probability_improvement);
    assert!(best_like.expected_improvement >= bad_like.expected_improvement);
}

#[test]
fn scoring_uses_speed_weight_when_configured() {
    let mut config = config();
    config.sweep_quality_weight = 0.0;
    config.sweep_speed_weight = 1.0;
    config.sweep_exploration_weight = 0.0;
    let trials = [
        trial(candidate(4, 4), 4.0),
        trial(candidate(4, 8), 4.1),
        trial(candidate(16, 4), 4.2),
        trial(candidate(16, 8), 4.3),
    ];
    let analysis = super::analyze(&trials, &config);
    let small_batch = super::score_candidate(&analysis, &config, &candidate(4, 4));
    let large_batch = super::score_candidate(&analysis, &config, &candidate(16, 4));

    assert!(large_batch.expected_speed > small_batch.expected_speed);
    assert!(large_batch.score > small_batch.score);
}

#[test]
fn factor_beliefs_aggregate_direction_and_confidence() {
    let config = config();
    let trials = [
        trial(candidate(4, 4), 9.0),
        trial(candidate(4, 8), 5.0),
        trial(candidate(16, 4), 5.0),
        trial(candidate(16, 8), 1.0),
    ];
    let analysis = super::analyze(&trials, &config);
    let beliefs = super::factor_beliefs(&analysis, &config);
    let batch = beliefs
        .iter()
        .find(|belief| belief.factor == "batch_size")
        .unwrap();

    assert!(batch.direction > 0.0);
    assert!(batch.confidence > 0.0);
    assert!(batch.variance >= 0.0);
}

#[test]
fn factor_beliefs_include_speed_direction_when_weighted() {
    let mut config = config();
    config.sweep_quality_weight = 0.0;
    config.sweep_speed_weight = 1.0;
    config.sweep_exploration_weight = 0.0;
    let trials = [
        trial(candidate(4, 4), 4.0),
        trial(candidate(4, 8), 4.1),
        trial(candidate(16, 4), 4.2),
        trial(candidate(16, 8), 4.3),
    ];
    let analysis = super::analyze(&trials, &config);
    let beliefs = super::factor_beliefs(&analysis, &config);
    let batch = beliefs
        .iter()
        .find(|belief| belief.factor == "batch_size")
        .unwrap();

    assert!(batch.direction > 0.0);
    assert!(batch.confidence > 0.0);
}

#[test]
fn stability_beliefs_do_not_create_target_direction() {
    let mut config = config();
    config.sweep_stability_weight = 1.0;
    let trials = [
        trial_with_status(candidate(4, 4), "failed_build"),
        trial_with_status(candidate(4, 8), "failed_build"),
        trial_with_status(candidate(16, 4), "success"),
        trial_with_status(candidate(16, 8), "success"),
    ];
    let analysis = super::analyze(&trials, &config);
    let beliefs = super::factor_beliefs(&analysis, &config);

    assert!(!beliefs.is_empty());
    assert!(
        beliefs
            .iter()
            .all(|belief| belief.direction.abs() < 1.0e-12)
    );
    assert!(beliefs.iter().any(|belief| belief.variance > 0.0));
}

#[test]
fn scoring_uses_stability_prior_when_stability_model_is_constant_failure() {
    let config = config();
    let trials = [
        trial_with_status(candidate(4, 4), "failed_build"),
        trial_with_status(candidate(4, 8), "failed_build"),
        trial_with_status(candidate(16, 4), "failed_run"),
        trial_with_status(candidate(16, 8), "nan"),
    ];
    let analysis = super::analyze(&trials, &config);
    let score = super::score_candidate(&analysis, &config, &candidate(8, 4));

    assert!(
        analysis
            .models
            .iter()
            .all(|model| model.name != "stability")
    );
    assert!(score.survival_prior < 0.5);
    assert!(score.expected_quality < -3.0);
}

fn trial(candidate: Candidate, val_loss: f64) -> Trial {
    Trial {
        candidate,
        status: "success".to_string(),
        val_loss: Some(val_loss),
        completed_steps: Some(10),
        elapsed_s: Some(5.0),
        screen_val_loss: Some(val_loss + 1.0),
        screen_completed_steps: Some(10),
        screen_elapsed_s: Some(5.0),
        screen_reason: Some("screen_loss_improved".to_string()),
        log_path: PathBuf::from("train.log"),
    }
}

fn trial_with_status(candidate: Candidate, status: &str) -> Trial {
    Trial {
        candidate,
        status: status.to_string(),
        val_loss: None,
        completed_steps: Some(10),
        elapsed_s: Some(5.0),
        screen_val_loss: None,
        screen_completed_steps: None,
        screen_elapsed_s: None,
        screen_reason: None,
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
        candidate_samples: 16,
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
