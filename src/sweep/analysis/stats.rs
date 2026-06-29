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

pub fn normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

pub fn normal_cdf(x: f64) -> f64 {
    let x = x.clamp(-8.0, 8.0);
    let inner = 0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x);
    (0.5 * (1.0 + inner.tanh())).clamp(0.0, 1.0)
}
