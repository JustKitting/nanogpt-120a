use super::super::snapshot::AdamSnapshot;

pub(super) fn adam_predicted_next(
    current: f32,
    grad: f32,
    first: f32,
    second: f32,
    config: &AdamSnapshot,
) -> f32 {
    let m = config.beta1 * first + (1.0 - config.beta1) * grad;
    let v = config.beta2 * second + (1.0 - config.beta2) * grad * grad;
    let update =
        (m / config.beta1_correction) / ((v / config.beta2_correction).sqrt() + config.eps);
    let decay = 1.0 - config.learning_rate * config.weight_decay;
    current * decay - config.learning_rate * update
}
