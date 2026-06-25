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
    let update = stabilized_gram_ns(polar, rows.min(cols), rows.max(cols), iterations, &[2]);
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

pub fn standard_polar(mut x: Vec<f32>, rows: usize, cols: usize, iterations: usize) -> Vec<f32> {
    for iter in 0..iterations {
        let gram = matmul(&x, &x, rows, rows, cols, true);
        let ax = matmul(&gram, &x, rows, cols, rows, false);
        let aax = matmul(&gram, &ax, rows, cols, rows, false);
        x = combine_next(&x, &ax, &aax, iter);
    }
    x
}

pub fn stabilized_gram_ns(
    mut x: Vec<f32>,
    rows: usize,
    cols: usize,
    iterations: usize,
    resets: &[usize],
) -> Vec<f32> {
    let mut r = matmul(&x, &x, rows, rows, cols, true);
    let mut q: Option<Vec<f32>> = None;

    for iter in 0..iterations {
        if iter != 0 && resets.contains(&iter) {
            x = matmul(
                q.as_ref().expect("restart needs accumulated Q"),
                &x,
                rows,
                cols,
                rows,
                false,
            );
            r = matmul(&x, &x, rows, rows, cols, true);
            q = None;
        }

        let (a, b, c) = coefficients(iter);
        let r2 = matmul(&r, &r, rows, rows, rows, false);
        let z = linear2(&r, b, &r2, c);

        q = Some(if q.is_none() {
            add_scaled_identity(&z, a, rows)
        } else {
            let q_ref = q.as_ref().expect("Q is set");
            let qz = matmul(q_ref, &z, rows, rows, rows, false);
            linear2(&qz, 1.0, q_ref, a)
        });

        if iter + 1 < iterations && !resets.contains(&(iter + 1)) {
            let rz_product = matmul(&r, &z, rows, rows, rows, false);
            let rz = linear2(&rz_product, 1.0, &r, a);
            let next_r_product = matmul(&z, &rz, rows, rows, rows, false);
            r = linear2(&next_r_product, 1.0, &rz, a);
        }
    }

    matmul(
        q.as_ref().expect("final Q is set"),
        &x,
        rows,
        cols,
        rows,
        false,
    )
}

pub fn normalized_polar_source(source: &[f32], rows: usize, cols: usize) -> Vec<f32> {
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

pub fn standard_cost(iterations: usize, aspect_ratio: usize) -> ProductCost {
    ProductCost {
        rectangular_products: iterations * 2,
        weighted_products: iterations * (2 * aspect_ratio + 1),
    }
}

pub fn stabilized_gram_ns_cost(
    iterations: usize,
    aspect_ratio: usize,
    resets: &[usize],
) -> ProductCost {
    let restart_count = resets
        .iter()
        .filter(|&&iter| iter != 0 && iter < iterations)
        .count();
    let rectangular_products = 1 + restart_count * 2 + 1;
    let weighted_products = 4 * iterations + 6 * aspect_ratio * restart_count - 6 * restart_count;

    ProductCost {
        rectangular_products,
        weighted_products,
    }
}

#[derive(Clone, Copy)]
pub struct ProductCost {
    pub rectangular_products: usize,
    pub weighted_products: usize,
}

pub fn gradient(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.0009 + ((i / cols) as f32) * 0.00002)
        .collect()
}

pub fn cosine(actual: &[f32], expected: &[f32]) -> f32 {
    let (dot, aa, bb) = actual
        .iter()
        .zip(expected)
        .fold((0.0, 0.0, 0.0), |(dot, aa, bb), (a, b)| {
            (a.mul_add(*b, dot), a.mul_add(*a, aa), b.mul_add(*b, bb))
        });
    dot / (aa.sqrt() * bb.sqrt())
}

pub fn relative_l2(actual: &[f32], expected: &[f32]) -> f32 {
    let (err, norm) = actual
        .iter()
        .zip(expected)
        .fold((0.0, 0.0), |(err, norm), (a, b)| {
            let diff = a - b;
            (diff.mul_add(diff, err), b.mul_add(*b, norm))
        });
    (err / norm).sqrt()
}

fn combine_next(x: &[f32], ax: &[f32], aax: &[f32], iter: usize) -> Vec<f32> {
    let (a, b, c) = coefficients(iter);
    x.iter()
        .zip(ax)
        .zip(aax)
        .map(|((x, ax), aax)| c.mul_add(*aax, a.mul_add(*x, b * *ax)))
        .collect()
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

fn linear2(a: &[f32], a_scale: f32, b: &[f32], b_scale: f32) -> Vec<f32> {
    a.iter()
        .zip(b)
        .map(|(a, b)| a_scale.mul_add(*a, b_scale * *b))
        .collect()
}

fn add_scaled_identity(x: &[f32], scale: f32, dim: usize) -> Vec<f32> {
    let mut out = x.to_vec();
    for i in 0..dim {
        out[i * dim + i] += scale;
    }
    out
}
