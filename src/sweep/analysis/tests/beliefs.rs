use crate::sweep::test_fixtures::{basic_candidate as candidate, trial_with_status};

use super::{config, quality_trials};

#[test]
fn factor_beliefs_aggregate_direction_and_confidence() {
    let config = config();
    let trials = quality_trials();
    let analysis = super::super::analyze(&trials, &config);
    let beliefs = super::super::factor_beliefs(&analysis, &config);
    let batch = beliefs
        .iter()
        .find(|belief| belief.factor == "batch_size")
        .unwrap();

    assert!(batch.direction > 0.0);
    assert!(batch.confidence > 0.0);
    assert!(batch.variance >= 0.0);
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
    let analysis = super::super::analyze(&trials, &config);
    let beliefs = super::super::factor_beliefs(&analysis, &config);

    assert!(!beliefs.is_empty());
    assert!(
        beliefs
            .iter()
            .all(|belief| belief.direction.abs() < 1.0e-12)
    );
    assert!(beliefs.iter().any(|belief| belief.variance > 0.0));
}
