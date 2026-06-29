use super::super::{candidate::Candidate, config::SweepConfig};
use super::{
    BinaryPrior, ResponseModel, SweepAnalysis,
    factors::{FEATURE_COUNT, candidate_features},
    regression::Prediction,
    stats::{EPS, normal_cdf, normal_pdf},
};

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
    let features = candidate_features(candidate);
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

fn improvement_acquisition(prediction: Option<(&ResponseModel, Prediction)>) -> (f64, f64) {
    let Some((model, prediction)) = prediction else {
        return (0.0, 0.0);
    };
    let sigma = prediction.uncertainty.max(EPS);
    let gap = prediction.standard_score - model.model.best_standard_score;
    let z = gap / sigma;
    let probability = normal_cdf(z);
    let expected = (gap * probability + sigma * normal_pdf(z)).max(0.0);
    (probability, expected)
}
