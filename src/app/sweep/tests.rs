use std::path::PathBuf;

use super::{
    analysis,
    baseline::Baseline,
    candidate::{Candidate, MIN_N_LAYER, valid_aurora_phases},
    chain,
    config::SweepConfig,
    history::History,
    history::Trial,
    parse::RunResult,
};

#[test]
fn exposes_profiled_l2_aurora_phase_layout() {
    assert!(valid_aurora_phases(8, 90).contains(&2));
    assert!(!valid_aurora_phases(16, 90).contains(&2));
    assert!(valid_aurora_phases(16, 90).contains(&4));
}

#[test]
fn starts_fresh_sweep_from_best_measured_baseline() {
    let measured = measured_candidate();
    let config = config(3, 32);
    let analysis = analysis::analyze(&[], &config);
    let baseline = super::optimizer::propose(
        &[],
        &Default::default(),
        &mut rng(),
        &config,
        &analysis,
        Some(&measured),
    )
    .candidate;

    assert_eq!(baseline.batch_size, 8);
    assert_eq!(baseline.n_layer, MIN_N_LAYER);
    assert_eq!(baseline.n_embd, 1024);
    assert_eq!(baseline.n_head, 16);
    assert_eq!(baseline.aurora_phases, 4);
    assert_eq!(baseline.aurora_blocks, 80);
    assert_eq!(baseline.lr_scale, 1.014_040);
    assert_eq!(baseline.adam_lr_scale, 1.980_467);
    assert_eq!(baseline.warmup_steps, 5);
    assert_eq!(baseline.start_ratio, 0.05);
    assert_eq!(baseline.amuse_beta1, 0.2);
    assert_eq!(baseline.amuse_rho, 0.5);
}

#[test]
fn measured_baseline_is_not_rerun_as_next_candidate() {
    let seed_history = [trial("success", Some(1.0), candidate(8, 2, 1.0))];
    let baseline_candidate = candidate(8, 4, 2.0);
    let baseline_trial = trial("success", Some(4.2), baseline_candidate.clone());
    let all_trials = chain::all_trials_with_baseline(Some(&baseline_trial), &seed_history, &[]);
    let seen = chain::seen_keys(&all_trials);
    let config = config(1, 16);
    let analysis = analysis::analyze(&all_trials, &config);
    let proposal = super::optimizer::propose(
        &all_trials,
        &seen,
        &mut rng(),
        &config,
        &analysis,
        Some(&baseline_candidate),
    );

    assert_ne!(proposal.candidate.key(), baseline_candidate.key());
    assert!(proposal.candidate.n_layer >= MIN_N_LAYER);
}

#[test]
fn random_candidates_respect_min_layer_count() {
    let mut rng = rng();
    for _ in 0..256 {
        assert!(Candidate::random(&mut rng).n_layer >= MIN_N_LAYER);
    }
}

#[test]
fn optimizer_ignores_sub_min_layer_history() {
    let trials = [trial("success", Some(1.0), candidate(8, 2, 1.0))];
    let config = config(3, 16);
    let analysis = analysis::analyze(&trials, &config);
    let proposal = super::optimizer::propose(
        &trials,
        &Default::default(),
        &mut rng(),
        &config,
        &analysis,
        None,
    );

    assert!(proposal.candidate.n_layer >= MIN_N_LAYER);
    assert_eq!(proposal.reason, "random");
    assert_eq!(proposal.ranked.len(), 1);
}

#[test]
fn failed_trials_count_toward_random_phase_progression() {
    let trials = [
        trial("failed_build", None, candidate(8, 4, 1.0)),
        trial("failed_run", None, candidate(16, 4, 1.2)),
    ];
    let config = config(2, 8);
    let analysis = analysis::analyze(&trials, &config);
    let proposal = super::optimizer::propose(
        &trials,
        &Default::default(),
        &mut rng(),
        &config,
        &analysis,
        None,
    );

    assert_eq!(proposal.reason, "model");
    assert_eq!(proposal.ranked.len(), config.candidate_samples);
}

#[test]
fn model_proposal_records_sorted_ranked_candidates() {
    let trials = [
        trial("success", Some(5.0), candidate(4, 4, 0.8)),
        trial("success", Some(4.0), candidate(8, 4, 1.0)),
        trial("success", Some(3.5), candidate(16, 8, 1.2)),
        trial("rejected_screen", None, candidate(4, 8, 2.0)),
    ];
    let config = config(0, 8);
    let analysis = analysis::analyze(&trials, &config);
    let proposal = super::optimizer::propose(
        &trials,
        &Default::default(),
        &mut rng(),
        &config,
        &analysis,
        None,
    );

    assert_eq!(proposal.reason, "model");
    assert_eq!(proposal.ranked.len(), config.candidate_samples);
    assert_eq!(proposal.candidate.key(), proposal.ranked[0].candidate.key());
    assert!(
        proposal
            .ranked
            .windows(2)
            .all(|pair| { pair[0].score.score >= pair[1].score.score })
    );
}

#[test]
fn promotes_baseline_file_when_validation_improves() {
    let path = temp_path("sweep-baseline.env");
    let mut baseline = Baseline::load(path.clone()).unwrap();

    assert!(
        baseline
            .promote_trial(&trial("success", Some(5.0), candidate(8, 4, 1.0)), false)
            .unwrap()
    );
    assert!(
        !baseline
            .promote_trial(&trial("success", Some(4.2), measured_candidate()), false)
            .unwrap()
    );
    assert!(
        baseline
            .promote_trial(&trial("success", Some(4.2), candidate(8, 4, 2.0)), false)
            .unwrap()
    );

    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("VAL_LOSS=4.200000"));
    assert!(text.contains("SCREEN_LOSS=5.200000"));
    assert!(text.contains("GPT2_BATCH_SIZE=8"));
    assert!(text.contains("GPT2_N_LAYER=4"));
    assert!(text.contains("GPT2_N_EMBD=1536"));
    assert!(text.contains("AURORA_MATRIX_PHASES=8"));
    assert!(text.contains("TRAIN_LR_SCALE=2.000000"));
    let loaded = Baseline::load(path.clone())
        .unwrap()
        .measured_trial()
        .unwrap();
    assert_eq!(loaded.screen_val_loss, Some(5.2));
    let _ = std::fs::remove_file(path);
}

#[test]
fn syncs_local_real_trials_to_shared_history_once() {
    let path = temp_path("sweep-shared-history.tsv");
    let mut shared = History::load(path.clone()).unwrap();
    let trial = trial("success", Some(4.2), candidate(8, 4, 1.0));

    chain::sync_shared_history(&mut shared, &[trial.clone()], false).unwrap();
    chain::sync_shared_history(&mut shared, &[trial], false).unwrap();

    let persisted = super::trial_row::read_trials(&path);
    assert_eq!(persisted.len(), 1);
    assert_eq!(persisted[0].val_loss, Some(4.2));
    let _ = std::fs::remove_file(path);
}

#[test]
fn chains_shared_and_local_trials_without_duplicates() {
    let shared = trial("success", Some(5.0), candidate(8, 4, 1.0));
    let local_new = trial("nan", None, candidate(8, 2, 2.0));
    let local_duplicate = shared.clone();

    let trials = chain::all_trials(&[shared], &[local_duplicate, local_new]);

    assert_eq!(trials.len(), 2);
    assert_eq!(trials[0].val_loss, Some(5.0));
    assert_eq!(trials[1].status, "nan");
}

#[test]
fn records_sweep_owned_trial_status_and_events() {
    let sweep_dir = temp_path("sweep-status");
    let trial_dir = sweep_dir.join("trial_0000");
    std::fs::create_dir_all(&trial_dir).unwrap();
    let candidate = measured_candidate();
    let result = RunResult {
        val_loss: Some(4.2),
        completed_steps: Some(128),
        last_step: Some(127),
        last_elapsed_s: Some(90.5),
        last_train_loss: Some(4.8),
        saw_nan: false,
    };

    super::status::record(&sweep_dir, &trial_dir, 0, &candidate, "success", &result).unwrap();

    let root_status = std::fs::read_to_string(sweep_dir.join("status.env")).unwrap();
    let trial_status = std::fs::read_to_string(trial_dir.join("status.env")).unwrap();
    let events = std::fs::read_to_string(sweep_dir.join("events.tsv")).unwrap();
    assert!(root_status.contains("EVENT=success"));
    assert!(root_status.contains("VAL_LOSS=4.200000"));
    assert!(trial_status.contains("COMPLETED_STEPS=128"));
    assert!(events.contains("success\t0\tb8_l2_d1024_h16_p2_c80_lr1.0140"));
    let _ = std::fs::remove_dir_all(sweep_dir);
}

fn trial(status: &str, val_loss: Option<f64>, candidate: Candidate) -> Trial {
    Trial {
        candidate,
        status: status.to_string(),
        val_loss,
        completed_steps: Some(10),
        elapsed_s: Some(5.0),
        screen_val_loss: val_loss.map(|loss| loss + 1.0),
        log_path: PathBuf::from("train.log"),
    }
}

fn candidate(batch_size: usize, n_layer: usize, lr_scale: f64) -> Candidate {
    Candidate {
        batch_size,
        n_layer,
        n_embd: 1536,
        n_head: 12,
        aurora_phases: 8,
        aurora_blocks: 180,
        lr_scale,
        adam_lr_scale: 1.0,
        warmup_steps: 5,
        start_ratio: 0.0,
        amuse_beta1: 0.4,
        amuse_rho: 0.8,
    }
}

fn config(random_trials: usize, candidate_samples: usize) -> SweepConfig {
    SweepConfig {
        trials: 4,
        random_trials,
        candidate_samples,
        max_seconds: 900.0,
        screen_steps: 500,
        screen_max_seconds: 180.0,
        sweep_quality_weight: 1.0,
        sweep_speed_weight: 0.25,
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

fn measured_candidate() -> Candidate {
    Candidate {
        batch_size: 8,
        n_layer: 2,
        n_embd: 1024,
        n_head: 16,
        aurora_phases: 2,
        aurora_blocks: 80,
        lr_scale: 1.014_040,
        adam_lr_scale: 1.980_467,
        warmup_steps: 5,
        start_ratio: 0.05,
        amuse_beta1: 0.2,
        amuse_rho: 0.5,
    }
}

fn temp_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("{}-{}-{name}", std::process::id(), nanos()));
    path
}

fn rng() -> super::rng::SweepRng {
    super::rng::SweepRng::new(0x4750_5432)
}

fn nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}
