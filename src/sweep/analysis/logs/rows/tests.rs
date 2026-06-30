use crate::sweep::analysis::logs::Observation;
use crate::sweep::test_fixtures::basic_candidate;

#[test]
fn screen_quality_uses_persisted_screen_loss_when_log_is_missing() {
    let rows = super::screen_quality_rows(
        &[Observation {
            candidate: candidate(),
            status: "rejected_screen".to_string(),
            screen: None,
            full: None,
            trial_val_loss: None,
            trial_elapsed_s: None,
            trial_screen_val_loss: Some(5.25),
            trial_screen_elapsed_s: Some(90.0),
            trial_screen_reason: Some("screen_loss_worse".to_string()),
        }],
        90.0,
    );

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].1, -5.25);
}

#[test]
fn screen_quality_ignores_different_time_budget() {
    let rows = super::screen_quality_rows(
        &[Observation {
            candidate: candidate(),
            status: "rejected_screen".to_string(),
            screen: None,
            full: None,
            trial_val_loss: None,
            trial_elapsed_s: None,
            trial_screen_val_loss: Some(3.25),
            trial_screen_elapsed_s: Some(180.0),
            trial_screen_reason: Some("screen_loss_improved".to_string()),
        }],
        30.0,
    );

    assert!(rows.is_empty());
}

#[test]
fn stability_marks_missing_val_loss_screen_rejection_as_failure() {
    let rows = super::stability_rows(&[Observation {
        candidate: candidate(),
        status: "rejected_screen".to_string(),
        screen: None,
        full: None,
        trial_val_loss: None,
        trial_elapsed_s: None,
        trial_screen_val_loss: None,
        trial_screen_elapsed_s: Some(180.0),
        trial_screen_reason: Some("missing_val_loss".to_string()),
    }]);

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].1, 0.0);
}

#[test]
fn stability_keeps_worse_screen_loss_as_survived() {
    let rows = super::stability_rows(&[Observation {
        candidate: candidate(),
        status: "rejected_screen".to_string(),
        screen: None,
        full: None,
        trial_val_loss: None,
        trial_elapsed_s: None,
        trial_screen_val_loss: Some(5.25),
        trial_screen_elapsed_s: Some(90.0),
        trial_screen_reason: Some("screen_loss_worse".to_string()),
    }]);

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].1, 1.0);
}

fn candidate() -> crate::sweep::candidate::Candidate {
    basic_candidate(8, 4)
}
