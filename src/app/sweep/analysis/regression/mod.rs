mod effects;
mod linear;

use super::super::candidate::Candidate;
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
    means: Vec<f64>,
    stds: Vec<f64>,
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

    let raw = rows
        .iter()
        .map(|(candidate, _)| candidate_features(candidate))
        .collect::<Vec<_>>();
    let (indices, means, stds) = active_features(&raw);
    if indices.is_empty() {
        return None;
    }

    let x = raw
        .iter()
        .map(|row| {
            indices
                .iter()
                .enumerate()
                .map(|(j, i)| (row[*i] - means[j]) / stds[j])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let yz = y.iter().map(|y| (y - y_mean) / y_std).collect::<Vec<_>>();
    let (beta, inverse) = linear::ridge_fit(&x, &yz)?;
    let residual_std = linear::residual_std(&x, &yz, &beta);
    let effects = effects::build(&indices, &beta, &inverse, residual_std);

    Some(Model {
        n: rows.len(),
        y_mean,
        y_std,
        residual_std,
        means,
        stds,
        indices,
        beta,
        covariance: inverse,
        effects,
    })
}

impl Model {
    pub fn predict(&self, features: &[f64; FEATURE_COUNT]) -> Prediction {
        let z = self
            .indices
            .iter()
            .enumerate()
            .map(|(j, i)| (features[*i] - self.means[j]) / self.stds[j])
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

fn active_features(raw: &[[f64; FEATURE_COUNT]]) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
    let mut indices = Vec::new();
    let mut means = Vec::new();
    let mut stds = Vec::new();
    for i in 0..FEATURE_COUNT {
        let values = raw.iter().map(|row| row[i]).collect::<Vec<_>>();
        let m = mean(&values);
        let s = stddev(&values, m);
        if s > EPS {
            indices.push(i);
            means.push(m);
            stds.push(s);
        }
    }
    (indices, means, stds)
}
