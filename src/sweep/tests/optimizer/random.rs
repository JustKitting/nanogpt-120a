use crate::sweep::candidate::MIN_N_LAYER;

use super::super::fixtures::{candidate, config, trial};
use super::propose;

#[test]
fn optimizer_ignores_sub_min_layer_history() {
    let trials = [trial("success", Some(1.0), candidate(8, 2, 1.0))];
    let config = config(3, 16);
    let proposal = propose(&trials, &config, None);

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
    let proposal = propose(&trials, &config, None);

    assert_eq!(proposal.reason, "model");
    assert_eq!(proposal.ranked.len(), config.candidate_samples);
}
