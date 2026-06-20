const NORM_SAFETY: f32 = 1.01;
const NORM_EPS: f32 = 1.0e-7;

use crate::polar_coefficients::coefficients;
use crate::polar_reference::round_f16_to_f32;

pub fn first_iteration_update(
    grad: &[f32],
    rows: usize,
    cols: usize,
    mu: f32,
    learning_rate: f32,
    weight_decay: f32,
    iterations: usize,
) -> Vec<f32> {
    let nesterov: Vec<f32> = grad
        .iter()
        .map(|g| (1.0 - mu).mul_add(*g, mu * (1.0 - mu) * *g))
        .collect();
    let polar = normalized_polar_source(&nesterov, rows, cols);
    let polar_rows = rows.min(cols);
    let polar_cols = rows.max(cols);
    let update = polar_iterations(polar, polar_rows, polar_cols, iterations);
    let scale = 0.2 * (rows.max(cols) as f32).sqrt();
    let decay = 1.0 - learning_rate * weight_decay;

    (0..rows * cols)
        .map(|index| {
            let update_index = if rows > cols {
                let row = index / cols;
                let col = index - row * cols;
                col * rows + row
            } else {
                index
            };
            decay - learning_rate * scale * update[update_index]
        })
        .collect()
}

fn polar_iterations(mut source: Vec<f32>, rows: usize, cols: usize, iterations: usize) -> Vec<f32> {
    for iter in 0..iterations {
        let gram = matmul(&source, &source, rows, rows, cols, true);
        let ax = matmul(&gram, &source, rows, cols, rows, false);
        let aax = matmul(&gram, &ax, rows, cols, rows, false);
        source = polar_next(&source, &ax, &aax, iter);
    }
    source
}

fn normalized_polar_source(source: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let inv_norm =
        1.0 / (source.iter().map(|v| v * v).sum::<f32>().sqrt() * NORM_SAFETY + NORM_EPS);
    if rows > cols {
        let mut out = vec![0.0; source.len()];
        for row in 0..rows {
            for col in 0..cols {
                out[col * rows + row] = source[row * cols + col] * inv_norm;
            }
        }
        out
    } else {
        source.iter().map(|v| v * inv_norm).collect()
    }
}

fn matmul(a: &[f32], b: &[f32], rows: usize, cols: usize, k_len: usize, rhs_t: bool) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0;
            for k in 0..k_len {
                let bv = if rhs_t {
                    b[col * k_len + k]
                } else {
                    b[k * cols + col]
                };
                sum += round_f16_to_f32(a[row * k_len + k]) * round_f16_to_f32(bv);
            }
            out[row * cols + col] = sum;
        }
    }
    out
}

fn polar_next(x: &[f32], ax: &[f32], aax: &[f32], iter: usize) -> Vec<f32> {
    let (a, b, c) = coefficients(iter);
    x.iter()
        .zip(ax)
        .zip(aax)
        .map(|((x, ax), aax)| c.mul_add(*aax, a.mul_add(*x, b * *ax)))
        .collect()
}
