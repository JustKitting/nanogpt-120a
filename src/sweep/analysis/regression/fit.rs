use crate::sweep::{candidate::Candidate, features::regression_features};

use super::super::{
    design,
    stats::{EPS, mean, stddev},
};
use super::{effects, features, linear, model::Model};

const MIN_ROWS: usize = 3;

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
    let best_value = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let best_standard_score = (best_value - y_mean) / y_std;

    let base_rows = rows
        .iter()
        .map(|(candidate, _)| regression_features(candidate))
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
        best_value,
        best_standard_score,
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
