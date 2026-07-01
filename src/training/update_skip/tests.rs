use super::{UpdateSkipConfig, UpdateSkipState};

fn state() -> UpdateSkipState {
    UpdateSkipState::from_config(UpdateSkipConfig::for_test())
}

#[test]
fn waits_for_minimum_history() {
    let mut state = state();
    assert!(!state.observe(Some(1.0), 1.0).skipped);
    assert!(!state.observe(Some(100.0), 100.0).skipped);
}

#[test]
fn skips_loss_outlier_after_history() {
    let mut state = state();
    assert!(!state.observe(Some(1.0), 1.0).skipped);
    assert!(!state.observe(Some(1.1), 1.0).skipped);
    let decision = state.observe(Some(10.0), 1.0);
    assert!(decision.skipped);
    assert!(decision.loss_spike);
    assert!(!decision.grad_norm_spike);
}

#[test]
fn skips_grad_norm_outlier_after_history() {
    let mut state = state();
    assert!(!state.observe(None, 1.0).skipped);
    assert!(!state.observe(None, 1.1).skipped);
    let decision = state.observe(None, 10.0);
    assert!(decision.skipped);
    assert!(!decision.loss_spike);
    assert!(decision.grad_norm_spike);
}

#[test]
fn skips_non_finite_without_history() {
    let mut state = state();
    let decision = state.observe(Some(f32::NAN), 1.0);
    assert!(decision.skipped);
    assert!(decision.non_finite);
}
