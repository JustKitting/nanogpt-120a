use super::super::{
    factors::FEATURE_NAMES,
    stats::{EPS, logistic},
};

#[derive(Clone, Debug)]
pub struct Effect {
    pub name: &'static str,
    pub coefficient: f64,
    pub stderr: f64,
    pub t: f64,
    pub p_positive: f64,
}

pub fn build(
    indices: &[usize],
    beta: &[f64],
    inverse: &[Vec<f64>],
    residual_std: f64,
) -> Vec<Effect> {
    let mut effects = indices
        .iter()
        .enumerate()
        .map(|(j, index)| {
            let stderr = (residual_std * residual_std * inverse[j][j].max(0.0)).sqrt();
            let t = if stderr > EPS { beta[j] / stderr } else { 0.0 };
            Effect {
                name: FEATURE_NAMES[*index],
                coefficient: beta[j],
                stderr,
                t,
                p_positive: logistic(1.702 * t),
            }
        })
        .collect::<Vec<_>>();
    effects.sort_by(|a, b| b.t.abs().total_cmp(&a.t.abs()));
    effects
}
