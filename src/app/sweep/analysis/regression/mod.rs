mod effects;
mod features;
mod linear;

use super::super::candidate::Candidate;
use super::design;
use super::factors::{FEATURE_COUNT, candidate_features};
use super::stats::{EPS, mean, stddev};

const MIN_ROWS: usize = 3;

#[derive(Clone, Copy, Debug)]
pub struct Prediction {
    pub value: f64,
    pub standard_score: f64,
    pub uncertainty: f64,
}

pub use effects::Effect;

#[derive(Clone, Debug)]
pub struct Model {
    pub n: usize,
    pub y_mean: f64,
    pub y_std: f64,
    pub residual_std: f64,
    base_stats: design::BaseStats,
    terms: Vec<design::Term>,
    design_means: Vec<f64>,
    design_stds: Vec<f64>,
    indices: Vec<usize>,
    beta: Vec<f64>,
    covariance: Vec<Vec<f64>>,
    pub effects: Vec<Effect>,
}

pub fn fit(rows: Vec<(Candidate, f64)>) -> Option<Model> {
    if rows.len() < MIN_ROWS {
        return None;
    }

    let y = rows.iter().map(|(_, y)| *y).collect::<Vec<_>>();
    let y_mean = mean(&y);
    let y_std = stddev(&y, y_mean);
    if y_std <= EPS {
        return None;
    }

    let base_rows = rows
        .iter()
        .map(|(candidate, _)| candidate_features(candidate))
        .collect::<Vec<_>>();
    let base_stats = design::base_stats(&base_rows);
    let terms = design::terms();
    let raw = base_rows
        .iter()
        .map(|row| design::values_from_base(row, &base_stats, &terms))
        .collect::<Vec<_>>();
    let names = terms
        .iter()
        .map(|term| design::term_name(*term))
        .collect::<Vec<_>>();
    let (indices, design_means, design_stds) = features::active(&raw);
    if indices.is_empty() {
        return None;
    }

    let x = raw
        .iter()
        .map(|row| {
            indices
                .iter()
                .enumerate()
                .map(|(j, i)| (row[*i] - design_means[j]) / design_stds[j])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let yz = y.iter().map(|y| (y - y_mean) / y_std).collect::<Vec<_>>();
    let (beta, inverse) = linear::ridge_fit(&x, &yz)?;
    let residual_std = linear::residual_std(&x, &yz, &beta);
    let effects = effects::build(&indices, &names, &beta, &inverse, residual_std);

    Some(Model {
        n: rows.len(),
        y_mean,
        y_std,
        residual_std,
        base_stats,
        terms,
        design_means,
        design_stds,
        indices,
        beta,
        covariance: inverse,
        effects,
    })
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
        let standard_score = super::stats::dot(&z, &self.beta);
        let variance = linear::quadratic_form(&z, &self.covariance).max(0.0);
        Prediction {
            value: self.y_mean + standard_score * self.y_std,
            standard_score,
            uncertainty: variance.sqrt(),
        }
    }
}
