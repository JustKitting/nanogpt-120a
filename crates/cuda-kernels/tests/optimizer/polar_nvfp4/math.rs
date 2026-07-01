use crate::polar_coefficients::coefficients;

pub use crate::polar_reference::{
    cosine, max_abs_error, normalized_polar_source as normalized_source,
    polar_iterations_f16 as polar_iterations_f16_leaf, relative_l2,
};

pub fn error_metrics(actual: &[f32], expected: &[f32]) -> (f32, f32, f32) {
    (
        cosine(actual, expected),
        relative_l2(actual, expected),
        max_abs_error(actual, expected),
    )
}

pub fn finite_error_metrics(actual: &[f32], expected: &[f32], finite: bool) -> (f32, f32, f32) {
    if finite {
        error_metrics(actual, expected)
    } else {
        (cosine(actual, expected), f32::INFINITY, f32::INFINITY)
    }
}

pub fn gradient(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| ((i % 41) as f32 - 20.0) * 0.0007 + ((i / cols) as f32) * 0.00003)
        .collect()
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
    crate::polar_reference::polar_next(x, ax, aax, iter)
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
    crate::polar_reference::matmul_f16(a, b_t, rows, cols, k_len, true)
}
