use std::path::PathBuf;

use super::{
    baseline::Baseline,
    candidate::{Candidate, valid_aurora_phases},
    chain,
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
    let baseline =
        super::optimizer::propose(&[], &Default::default(), &mut rng(), 3, 32, Some(&measured));

    assert_eq!(baseline.batch_size, 8);
    assert_eq!(baseline.n_layer, 2);
    assert_eq!(baseline.n_embd, 1024);
    assert_eq!(baseline.n_head, 16);
    assert_eq!(baseline.aurora_phases, 2);
    assert_eq!(baseline.aurora_blocks, 80);
    assert_eq!(baseline.lr_scale, 1.014_040);
    assert_eq!(baseline.adam_lr_scale, 1.980_467);
    assert_eq!(baseline.warmup_steps, 5);
    assert_eq!(baseline.start_ratio, 0.05);
    assert_eq!(baseline.amuse_beta1, 0.2);
    assert_eq!(baseline.amuse_rho, 0.5);
}

#[test]
fn promotes_baseline_file_when_validation_improves() {
    let path = temp_path("sweep-baseline.env");
    let mut baseline = Baseline::load(path.clone()).unwrap();

    assert!(
        baseline
            .promote_trial(&trial("success", Some(5.0), candidate(8, 2, 1.0)), false)
            .unwrap()
    );
    assert!(
        baseline
            .promote_trial(&trial("success", Some(4.2), measured_candidate()), false)
            .unwrap()
    );
    assert!(
        !baseline
            .promote_trial(&trial("success", Some(4.8), candidate(8, 4, 2.0)), false)
            .unwrap()
    );

    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("VAL_LOSS=4.200000"));
    assert!(text.contains("GPT2_BATCH_SIZE=8"));
    assert!(text.contains("GPT2_N_LAYER=2"));
    assert!(text.contains("GPT2_N_EMBD=1024"));
    assert!(text.contains("AURORA_MATRIX_PHASES=2"));
    assert!(text.contains("TRAIN_ADAM_LR_SCALE=1.980467"));
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
