use super::super::candidate::Candidate;
use super::factors::{FEATURE_COUNT, FEATURE_NAMES, candidate_features};
use super::stats::{EPS, logistic, mean, stddev};

const RIDGE_LAMBDA: f64 = 1.0;
const MIN_ROWS: usize = 4;

#[derive(Clone, Debug)]
pub struct InteractionEffect {
    pub name: String,
    pub coefficient: f64,
    pub stderr: f64,
    pub t: f64,
    pub p_positive: f64,
}

pub fn fit(rows: &[(Candidate, f64)]) -> Vec<InteractionEffect> {
    if rows.len() < MIN_ROWS {
        return Vec::new();
    }

    let y = rows.iter().map(|(_, y)| *y).collect::<Vec<_>>();
    let y_mean = mean(&y);
    let y_std = stddev(&y, y_mean);
    if y_std <= EPS {
        return Vec::new();
    }
    let yz = y.iter().map(|y| (y - y_mean) / y_std).collect::<Vec<_>>();
    let raw = rows
        .iter()
        .map(|(candidate, _)| candidate_features(candidate))
        .collect::<Vec<_>>();
    let stats = feature_stats(&raw);

    let mut effects = Vec::new();
    for left in 0..FEATURE_COUNT {
        for right in left + 1..FEATURE_COUNT {
            if let Some(effect) = interaction_effect(&raw, &yz, &stats, left, right) {
                effects.push(effect);
            }
        }
    }
    effects.sort_by(|a, b| b.t.abs().total_cmp(&a.t.abs()));
    effects
}

fn interaction_effect(
    raw: &[[f64; FEATURE_COUNT]],
    yz: &[f64],
    stats: &[(f64, f64); FEATURE_COUNT],
    left: usize,
    right: usize,
) -> Option<InteractionEffect> {
    let (left_mean, left_std) = stats[left];
    let (right_mean, right_std) = stats[right];
    if left_std <= EPS || right_std <= EPS {
        return None;
    }

    let product = raw
        .iter()
        .map(|row| ((row[left] - left_mean) / left_std) * ((row[right] - right_mean) / right_std))
        .collect::<Vec<_>>();
    let product_mean = mean(&product);
    let product_std = stddev(&product, product_mean);
    if product_std <= EPS {
        return None;
    }
    let x = product
        .iter()
        .map(|value| (value - product_mean) / product_std)
        .collect::<Vec<_>>();
    let denom = x.iter().map(|value| value * value).sum::<f64>() + RIDGE_LAMBDA;
    let coefficient = x.iter().zip(yz).map(|(x, y)| x * y).sum::<f64>() / denom;
    let residual_std = residual_std(&x, yz, coefficient);
    let stderr = (residual_std * residual_std / denom).sqrt();
    let t = if stderr > EPS {
        coefficient / stderr
    } else {
        0.0
    };

    Some(InteractionEffect {
        name: format!("{}*{}", FEATURE_NAMES[left], FEATURE_NAMES[right]),
        coefficient,
        stderr,
        t,
        p_positive: logistic(1.702 * t),
    })
}

fn feature_stats(raw: &[[f64; FEATURE_COUNT]]) -> [(f64, f64); FEATURE_COUNT] {
    std::array::from_fn(|i| {
        let values = raw.iter().map(|row| row[i]).collect::<Vec<_>>();
        let mean = mean(&values);
        (mean, stddev(&values, mean))
    })
}

fn residual_std(x: &[f64], y: &[f64], coefficient: f64) -> f64 {
    let rss = x
        .iter()
        .zip(y)
        .map(|(x, y)| {
            let err = y - x * coefficient;
            err * err
        })
        .sum::<f64>();
    (rss / x.len().saturating_sub(1).max(1) as f64).sqrt()
}
