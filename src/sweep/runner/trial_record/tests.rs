use std::{fs, path::PathBuf};

use crate::sweep::history::Trial;
use crate::sweep::test_fixtures::basic_candidate;

use super::{current_baseline_trial, promoted_screen_loss};

#[test]
fn reads_promoted_screen_loss_from_decision_artifact() {
    let dir = std::env::temp_dir().join(format!("sweep-screen-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("screen_decision.env"), "SCREEN_LOSS=3.250000\n").unwrap();
    let trial = Trial {
        candidate: candidate(),
        status: "success".to_string(),
        val_loss: Some(3.0),
        completed_steps: Some(100),
        elapsed_s: Some(900.0),
        screen_val_loss: Some(3.25),
        screen_completed_steps: Some(500),
        screen_elapsed_s: Some(90.0),
        screen_reason: Some("screen_loss_improved".to_string()),
        log_path: dir.join("train.log"),
    };

    assert_eq!(promoted_screen_loss(&trial), Some(3.25));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn current_screened_baseline_replaces_stale_baseline_screen_metrics() {
    let measured = Trial {
        candidate: candidate(),
        status: "success".to_string(),
        val_loss: Some(3.5),
        completed_steps: Some(3000),
        elapsed_s: Some(900.0),
        screen_val_loss: Some(6.9),
        screen_completed_steps: Some(500),
        screen_elapsed_s: Some(180.0),
        screen_reason: Some("old_screen".to_string()),
        log_path: PathBuf::from("old/train.log"),
    };
    let screened = Trial {
        screen_val_loss: Some(6.2),
        screen_completed_steps: Some(100),
        screen_elapsed_s: Some(30.0),
        screen_reason: Some("screen_baseline".to_string()),
        log_path: PathBuf::from("screen_baseline/screen.log"),
        ..measured.clone()
    };

    let current = current_baseline_trial(Some(&screened), Some(measured)).unwrap();

    assert_eq!(current.screen_val_loss, Some(6.2));
    assert_eq!(current.screen_elapsed_s, Some(30.0));
    assert_eq!(
        current.log_path,
        PathBuf::from("screen_baseline/screen.log")
    );
}

fn candidate() -> crate::sweep::candidate::Candidate {
    basic_candidate(8, 4)
}
