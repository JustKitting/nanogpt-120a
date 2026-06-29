use super::super::stats::{EPS, mean, stddev};

pub fn active(raw: &[Vec<f64>]) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
    let mut indices = Vec::new();
    let mut means = Vec::new();
    let mut stds = Vec::new();
    for i in 0..raw.first().map(Vec::len).unwrap_or(0) {
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
