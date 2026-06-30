use crate::sweep::test_fixtures::{
    basic_candidate as candidate, trial_with_losses, trial_with_status,
};

use super::{config, has_model, quality_trials};

#[test]
fn scoring_uses_pairwise_interaction_signal() {
    let config = config();
    let trials = [
        super::trial(candidate(4, 4), 1.0),
        super::trial(candidate(16, 8), 1.0),
        super::trial(candidate(4, 8), 9.0),
        super::trial(candidate(16, 4), 9.0),
    ];
    let analysis = super::super::analyze(&trials, &config);
    let aligned = super::super::score_candidate(&analysis, &config, &candidate(16, 8));
    let crossed = super::super::score_candidate(&analysis, &config, &candidate(16, 4));

    let aligned_quality = aligned.predicted_quality.unwrap().standard_score;
    let crossed_quality = crossed.predicted_quality.unwrap().standard_score;
    assert!(aligned_quality > crossed_quality + 0.5);
    assert!(analysis.models.iter().any(|model| {
        model.name == "full_quality"
            && model
                .model
                .effects
                .iter()
                .any(|effect| effect.name == "batch_size*n_layer")
    }));
}

#[test]
fn scoring_reports_expected_improvement_against_best_observed_quality() {
    let config = config();
    let trials = quality_trials();
    let analysis = super::super::analyze(&trials, &config);
    let best_like = super::super::score_candidate(&analysis, &config, &candidate(16, 8));
    let bad_like = super::super::score_candidate(&analysis, &config, &candidate(4, 4));

    assert!((0.0..=1.0).contains(&best_like.probability_improvement));
    assert!(best_like.expected_improvement.is_finite());
    assert!(best_like.probability_improvement > bad_like.probability_improvement);
    assert!(best_like.expected_improvement >= bad_like.expected_improvement);
}

#[test]
fn scoring_prefers_screen_quality_over_sparse_full_quality_for_proposals() {
    let config = config();
    let trials = [
        trial_with_losses(candidate(4, 4), 9.0, 1.0),
        trial_with_losses(candidate(8, 4), 7.0, 3.0),
        trial_with_losses(candidate(16, 4), 3.0, 7.0),
        trial_with_losses(candidate(32, 4), 1.0, 9.0),
    ];
    let analysis = super::super::analyze(&trials, &config);
    let low_batch = super::super::score_candidate(&analysis, &config, &candidate(4, 4));
    let high_batch = super::super::score_candidate(&analysis, &config, &candidate(32, 4));

    assert!(has_model(&analysis, "screen_quality"));
    assert!(has_model(&analysis, "full_quality"));
    assert!(
        low_batch.predicted_quality.unwrap().standard_score
            > high_batch.predicted_quality.unwrap().standard_score
    );
}

#[test]
fn scoring_uses_stability_prior_when_stability_model_is_constant_failure() {
    let config = config();
    let trials = [
        trial_with_status(candidate(4, 4), "failed_build"),
        trial_with_status(candidate(4, 8), "failed_build"),
        trial_with_status(candidate(16, 4), "failed_run"),
        trial_with_status(candidate(16, 8), "nan"),
    ];
    let analysis = super::super::analyze(&trials, &config);
    let score = super::super::score_candidate(&analysis, &config, &candidate(8, 4));

    assert!(!has_model(&analysis, "stability"));
    assert!(score.survival_prior < 0.5);
    assert!(score.expected_quality < -3.0);
}
