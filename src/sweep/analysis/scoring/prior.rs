use super::super::{BinaryPrior, ResponseModel, regression::Prediction};

pub(super) fn survival_prior(
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
