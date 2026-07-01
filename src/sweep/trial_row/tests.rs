use std::path::PathBuf;

use crate::sweep::{
    history::Trial,
    test_fixtures::{basic_candidate, success_trial},
};

use super::{format::format_trial, parse::parse_trial};

#[test]
fn roundtrips_elapsed_time_in_new_rows() {
    let trial = Trial {
        completed_steps: Some(512),
        log_path: PathBuf::from("target/train.log"),
        elapsed_s: Some(123.5),
        screen_val_loss: Some(6.25),
        screen_completed_steps: Some(500),
        screen_elapsed_s: Some(90.25),
        screen_reason: Some("screen_loss_improved".to_string()),
        ..success_trial(candidate(), 4.25)
    };

    let parsed = parse_trial(&format_trial(&trial)).unwrap();
    assert_eq!(parsed.elapsed_s, Some(123.5));
    assert_eq!(parsed.screen_val_loss, Some(6.25));
    assert_eq!(parsed.screen_completed_steps, Some(500));
    assert_eq!(parsed.screen_elapsed_s, Some(90.25));
    assert_eq!(
        parsed.screen_reason.as_deref(),
        Some("screen_loss_improved")
    );
    assert_eq!(parsed.completed_steps, Some(512));
    assert_eq!(parsed.candidate.key(), trial.candidate.key());
}

#[test]
fn parses_old_rows_without_elapsed_time() {
    let parsed = parse_trial(
        "success\t4.250000\t512\t8\t4\t1024\t16\t4\t80\t1.000000\t1.000000\t20\t0.100000\t0.400000\t0.800000\ttarget/train.log",
    )
    .unwrap();

    assert_eq!(parsed.elapsed_s, None);
    assert_eq!(parsed.screen_val_loss, None);
    assert_eq!(parsed.screen_completed_steps, None);
    assert_eq!(parsed.screen_elapsed_s, None);
    assert_eq!(parsed.screen_reason, None);
    assert_eq!(parsed.completed_steps, Some(512));
    assert_eq!(parsed.candidate.batch_size, 8);
}

fn candidate() -> crate::sweep::candidate::Candidate {
    basic_candidate(8, 4)
}
