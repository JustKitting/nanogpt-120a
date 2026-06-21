use super::{CandidateScore, RunResult, decide};

#[test]
fn model_prior_can_pass_worse_screen_loss() {
    let result = RunResult {
        val_loss: Some(4.2),
        completed_steps: Some(500),
        ..RunResult::default()
    };
    let score = score(0.7, 0.8);
    let decision = decide(&result, Some(4.0), 500, Some(&score));

    assert!(decision.pass);
    assert_eq!(decision.reason, "model_expected_improvement");
}

#[test]
fn unstable_prior_does_not_pass_worse_screen_loss() {
    let result = RunResult {
        val_loss: Some(4.2),
        completed_steps: Some(500),
        ..RunResult::default()
    };
    let score = score(0.7, 0.2);
    let decision = decide(&result, Some(4.0), 500, Some(&score));

    assert!(!decision.pass);
}

fn score(expected_quality: f64, survival_prior: f64) -> CandidateScore {
    CandidateScore {
        score: expected_quality,
        expected_quality,
        survival_prior,
        expected_speed: 0.0,
        probability_improvement: 0.0,
        expected_improvement: 0.0,
        uncertainty: 0.0,
        exploration: 0.0,
        predicted_quality: None,
        predicted_speed: None,
        predicted_stability: None,
    }
}
