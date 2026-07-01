use super::super::features::{regression_features, FEATURE_COUNT};
use super::super::{candidate::Candidate, config::SweepConfig};
use super::{regression::Prediction, ResponseModel, SweepAnalysis};

mod acquisition;
mod prior;

use acquisition::improvement_acquisition;
use prior::survival_prior;

#[derive(Clone, Debug)]
pub struct CandidateScore {
    pub score: f64,
    pub expected_quality: f64,
    pub survival_prior: f64,
    pub probability_improvement: f64,
    pub expected_improvement: f64,
    pub uncertainty: f64,
    pub exploration: f64,
    pub predicted_quality: Option<Prediction>,
    pub predicted_stability: Option<Prediction>,
}

pub fn score_candidate(
    analysis: &SweepAnalysis,
    config: &SweepConfig,
    candidate: &Candidate,
) -> CandidateScore {
    let features = regression_features(candidate);
    let predicted_quality_model =
        best_prediction(analysis, &features, &["screen_quality", "full_quality"]);
    let predicted_quality = predicted_quality_model.map(|(_, prediction)| prediction);
    let predicted_stability_model = best_prediction(analysis, &features, &["stability"]);
    let predicted_stability = predicted_stability_model.map(|(_, prediction)| prediction);

    let quality = predicted_quality.map(|p| p.standard_score).unwrap_or(0.0);
    let (probability_improvement, expected_improvement) =
        improvement_acquisition(predicted_quality_model);
    let survival_prior = survival_prior(analysis.stability_prior, predicted_stability_model);
    let expected_quality =
        survival_prior * (quality + expected_improvement) + (1.0 - survival_prior) * -6.0;
    let uncertainty = [predicted_quality, predicted_stability]
        .into_iter()
        .flatten()
        .map(|p| p.uncertainty)
        .fold(0.0, f64::max);
    let exploration = uncertainty.ln_1p();
    let score = config.sweep_quality_weight * expected_quality
        + config.sweep_exploration_weight * exploration;

    CandidateScore {
        score,
        expected_quality,
        survival_prior,
        probability_improvement,
        expected_improvement,
        uncertainty,
        exploration,
        predicted_quality,
        predicted_stability,
    }
}

fn best_prediction<'a>(
    analysis: &'a SweepAnalysis,
    features: &[f64; FEATURE_COUNT],
    names: &[&str],
) -> Option<(&'a ResponseModel, Prediction)> {
    names.iter().find_map(|name| {
        analysis
            .models
            .iter()
            .find(|model| model.name == *name)
            .map(|model| (model, model.model.predict(features)))
    })
}
