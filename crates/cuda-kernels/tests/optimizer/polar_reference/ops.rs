use crate::polar_coefficients::coefficients;

use super::round_f16_to_f32;

pub fn polar_next(x: &[f32], ax: &[f32], aax: &[f32], iter: usize) -> Vec<f32> {
    let (a, b, c) = coefficients(iter);
    x.iter()
        .zip(ax)
        .zip(aax)
        .map(|((x, ax), aax)| c.mul_add(*aax, a.mul_add(*x, b * *ax)))
        .collect()
}

pub fn polar_iterations_f16(mut x: Vec<f32>, rows: usize, cols: usize, n: usize) -> Vec<f32> {
    for iter in 0..n {
        let gram = matmul_f16(&x, &x, rows, rows, cols, true);
        let ax = matmul_f16(&gram, &x, rows, cols, rows, false);
        let aax = matmul_f16(&gram, &ax, rows, cols, rows, false);
        x = polar_next(&x, &ax, &aax, iter);
    }
    x
}

pub fn matmul_f16(
    a: &[f32],
    b: &[f32],
    rows: usize,
    cols: usize,
    k_len: usize,
    rhs_t: bool,
) -> Vec<f32> {
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
