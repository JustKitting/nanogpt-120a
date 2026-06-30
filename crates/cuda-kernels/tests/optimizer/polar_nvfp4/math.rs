use crate::polar_coefficients::coefficients;
use crate::polar_reference::round_f16_to_f32;

pub use crate::polar_reference::{cosine, max_abs_error, relative_l2};

pub fn gradient(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| ((i % 41) as f32 - 20.0) * 0.0007 + ((i / cols) as f32) * 0.00003)
        .collect()
}

pub fn normalized_source(source: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let inv_norm = 1.0 / (source.iter().map(|v| v * v).sum::<f32>().sqrt() * 1.01 + 1.0e-7);
    if rows > cols {
        transpose_scaled(source, rows, cols, inv_norm)
    } else {
        source.iter().map(|v| v * inv_norm).collect()
    }
}

pub fn polar_iterations_f16_leaf(
    mut source: Vec<f32>,
    rows: usize,
    cols: usize,
    iterations: usize,
) -> Vec<f32> {
    for iter in 0..iterations {
        source = polar_step_f16_leaf(&source, rows, cols, iter);
    }
    source
}

pub fn gram_form_polar_iterations_f16_leaf(
    mut source: Vec<f32>,
    rows: usize,
    cols: usize,
    iterations: usize,
) -> Vec<f32> {
    for iter in 0..iterations {
        source = gram_form_polar_step_f16_leaf(&source, rows, cols, iter);
    }
    source
}

pub fn polar_step_f16_leaf(source: &[f32], rows: usize, cols: usize, iter: usize) -> Vec<f32> {
    let gram = matmul_f16_leaf(source, source, rows, rows, cols);
    let ax = matmul_f16_leaf(&gram, &transpose(source, rows, cols), rows, cols, rows);
    let aax = matmul_f16_leaf(&gram, &transpose(&ax, rows, cols), rows, cols, rows);
    combine_next(source, &ax, &aax, iter)
}

pub fn gram_form_polar_step_f16_leaf(
    source: &[f32],
    rows: usize,
    cols: usize,
    iter: usize,
) -> Vec<f32> {
    let (a, b, c) = coefficients(iter);
    let gram = matmul_f16_leaf(source, source, rows, rows, cols);
    let gram2 = matmul_f16_leaf(&gram, &transpose(&gram, rows, rows), rows, rows, rows);
    let mut q = gram
        .iter()
        .zip(gram2)
        .map(|(gram, gram2)| c.mul_add(gram2, b * *gram))
        .collect::<Vec<_>>();
    for row in 0..rows {
        q[row * rows + row] += a;
    }
    matmul_f16_leaf(&q, &transpose(source, rows, cols), rows, cols, rows)
}

pub fn combine_next(x: &[f32], ax: &[f32], aax: &[f32], iter: usize) -> Vec<f32> {
    let (a, b, c) = coefficients(iter);
    x.iter()
        .zip(ax)
        .zip(aax)
        .map(|((x, ax), aax)| c.mul_add(*aax, a.mul_add(*x, b * *ax)))
        .collect()
}

pub fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = x[row * cols + col];
        }
    }
    out
}

pub fn matmul_f16_leaf(a: &[f32], b_t: &[f32], rows: usize, cols: usize, k_len: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0;
            for k in 0..k_len {
                sum +=
                    round_f16_to_f32(a[row * k_len + k]) * round_f16_to_f32(b_t[col * k_len + k]);
            }
            out[row * cols + col] = sum;
        }
    }
    out
}

fn transpose_scaled(x: &[f32], rows: usize, cols: usize, scale: f32) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = x[row * cols + col] * scale;
        }
    }
    out
}
