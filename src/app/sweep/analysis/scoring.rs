use super::super::{candidate::Candidate, config::SweepConfig};
use super::{
    SweepAnalysis,
    factors::{FEATURE_COUNT, candidate_features},
    regression::Prediction,
};

#[derive(Clone, Debug)]
pub struct CandidateScore {
    pub score: f64,
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
        best_prediction(analysis, &features, &["full_quality", "screen_quality"]);
    let predicted_speed = best_prediction(
        analysis,
        &features,
        &["full_tokens_per_s", "screen_tokens_per_s"],
    );
    let predicted_stability = best_prediction(analysis, &features, &["stability"]);

    let quality = predicted_quality.map(|p| p.standard_score).unwrap_or(0.0);
    let speed = predicted_speed.map(|p| p.standard_score).unwrap_or(0.0);
    let stability = predicted_stability.map(|p| p.standard_score).unwrap_or(0.0);
    let uncertainty = [predicted_quality, predicted_speed, predicted_stability]
        .into_iter()
        .flatten()
        .map(|p| p.uncertainty)
        .fold(0.0, f64::max);
    let score = config.sweep_quality_weight * quality
        + config.sweep_speed_weight * speed
        + config.sweep_stability_weight * stability
        + config.sweep_exploration_weight * uncertainty;

    CandidateScore {
        score,
        predicted_quality,
        predicted_speed,
        predicted_stability,
    }
}

fn best_prediction(
    analysis: &SweepAnalysis,
    features: &[f64; FEATURE_COUNT],
    names: &[&str],
) -> Option<Prediction> {
    names.iter().find_map(|name| {
        analysis
            .models
            .iter()
            .find(|model| model.name == *name)
            .map(|model| model.model.predict(features))
    })
}
