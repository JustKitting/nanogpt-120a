use super::super::{candidate::Candidate, config::SweepConfig};
use super::{
    BinaryPrior, ResponseModel, SweepAnalysis,
    factors::{FEATURE_COUNT, candidate_features},
    regression::Prediction,
};

#[derive(Clone, Debug)]
pub struct CandidateScore {
    pub score: f64,
    pub expected_quality: f64,
    pub survival_prior: f64,
    pub uncertainty: f64,
    pub exploration: f64,
    pub predicted_quality: Option<Prediction>,
    pub predicted_speed: Option<Prediction>,
    pub predicted_stability: Option<Prediction>,
}

pub fn score_candidate(
    analysis: &SweepAnalysis,
    config: &SweepConfig,
    candidate: &Candidate,
) -> CandidateScore {
    let features = candidate_features(candidate);
    let predicted_quality =
        best_prediction(analysis, &features, &["full_quality", "screen_quality"])
            .map(|(_, prediction)| prediction);
    let predicted_speed = best_prediction(
        analysis,
        &features,
        &["full_tokens_per_s", "screen_tokens_per_s"],
    )
    .map(|(_, prediction)| prediction);
    let predicted_stability_model = best_prediction(analysis, &features, &["stability"]);
    let predicted_stability = predicted_stability_model.map(|(_, prediction)| prediction);

    let quality = predicted_quality.map(|p| p.standard_score).unwrap_or(0.0);
    let survival_prior = survival_prior(analysis.stability_prior, predicted_stability_model);
    let expected_quality = survival_prior * quality + (1.0 - survival_prior) * -6.0;
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
        uncertainty,
        exploration,
        predicted_quality,
        predicted_speed,
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

fn survival_prior(
    base: Option<BinaryPrior>,
    prediction: Option<(&ResponseModel, Prediction)>,
) -> f64 {
    let Some(base) = base else {
        return 1.0;
    };
    let base_rate = base.posterior_mean.clamp(0.01, 1.0);
    let Some((model, prediction)) = prediction else {
        return base_rate;
    };

    let predicted = prediction.value.clamp(0.01, 1.0);
    let sample_confidence = (model.model.n as f64 / (model.model.n as f64 + 8.0)).clamp(0.0, 1.0);
    let uncertainty_confidence = 1.0 / (1.0 + prediction.uncertainty.max(0.0));
    let confidence = (sample_confidence * uncertainty_confidence).clamp(0.0, 1.0);
    (base_rate * (1.0 - confidence) + predicted * confidence).clamp(0.01, 1.0)
}
