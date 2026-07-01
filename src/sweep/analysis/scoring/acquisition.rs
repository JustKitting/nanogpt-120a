use super::super::{
    ResponseModel,
    regression::Prediction,
    stats::{EPS, normal_cdf, normal_pdf},
};

pub(super) fn improvement_acquisition(
    prediction: Option<(&ResponseModel, Prediction)>,
) -> (f64, f64) {
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
