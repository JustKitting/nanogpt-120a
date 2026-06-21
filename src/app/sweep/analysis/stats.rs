pub const EPS: f64 = 1.0e-12;

pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

pub fn stddev(values: &[f64], mean: f64) -> f64 {
    (values
        .iter()
        .map(|value| {
            let d = value - mean;
            d * d
        })
        .sum::<f64>()
        / values.len().saturating_sub(1).max(1) as f64)
        .sqrt()
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

pub fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-32.0, 32.0)).exp())
}
