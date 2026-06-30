use crate::sweep::features::FEATURE_COUNT;

use super::super::{design, stats};
use super::{effects::Effect, linear};

#[derive(Clone, Copy, Debug)]
pub struct Prediction {
    pub value: f64,
    pub standard_score: f64,
    pub uncertainty: f64,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub n: usize,
    pub y_mean: f64,
    pub y_std: f64,
    pub best_value: f64,
    pub best_standard_score: f64,
    pub residual_std: f64,
    pub(super) base_stats: design::BaseStats,
    pub(super) terms: Vec<design::Term>,
    pub(super) design_means: Vec<f64>,
    pub(super) design_stds: Vec<f64>,
    pub(super) indices: Vec<usize>,
    pub(super) beta: Vec<f64>,
    pub(super) covariance: Vec<Vec<f64>>,
    pub effects: Vec<Effect>,
}

impl Model {
    pub fn predict(&self, features: &[f64; FEATURE_COUNT]) -> Prediction {
        let values = design::values_from_base(features, &self.base_stats, &self.terms);
        let z = self
            .indices
            .iter()
            .enumerate()
            .map(|(j, i)| (values[*i] - self.design_means[j]) / self.design_stds[j])
            .collect::<Vec<_>>();
        let standard_score = stats::dot(&z, &self.beta);
        let leverage = linear::quadratic_form(&z, &self.covariance).max(0.0);
        Prediction {
            value: self.y_mean + standard_score * self.y_std,
            standard_score,
            uncertainty: self.residual_std * (1.0 + leverage).sqrt(),
        }
    }
}
