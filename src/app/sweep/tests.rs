use std::path::PathBuf;

use super::{candidate::Candidate, chain, history::History, history::Trial};

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

fn temp_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("{}-{}-{name}", std::process::id(), nanos()));
    path
}

fn nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}
