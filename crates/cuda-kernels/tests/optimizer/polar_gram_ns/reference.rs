use crate::polar_coefficients::coefficients;
use crate::polar_reference::{matmul_f16, polar_next};

pub use crate::polar_reference::{cosine, normalized_polar_source, relative_l2};

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
        let gram = matmul_f16(&x, &x, rows, rows, cols, true);
        let ax = matmul_f16(&gram, &x, rows, cols, rows, false);
        let aax = matmul_f16(&gram, &ax, rows, cols, rows, false);
        x = polar_next(&x, &ax, &aax, iter);
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
    let mut r = matmul_f16(&x, &x, rows, rows, cols, true);
    let mut q: Option<Vec<f32>> = None;

    for iter in 0..iterations {
        if iter != 0 && resets.contains(&iter) {
            x = matmul_f16(
                q.as_ref().expect("restart needs accumulated Q"),
                &x,
                rows,
                cols,
                rows,
                false,
            );
            r = matmul_f16(&x, &x, rows, rows, cols, true);
            q = None;
        }

        let (a, b, c) = coefficients(iter);
        let r2 = matmul_f16(&r, &r, rows, rows, rows, false);
        let z = linear2(&r, b, &r2, c);

        q = Some(if q.is_none() {
            add_scaled_identity(&z, a, rows)
        } else {
            let q_ref = q.as_ref().expect("Q is set");
            let qz = matmul_f16(q_ref, &z, rows, rows, rows, false);
            linear2(&qz, 1.0, q_ref, a)
        });

        if iter + 1 < iterations && !resets.contains(&(iter + 1)) {
            let rz_product = matmul_f16(&r, &z, rows, rows, rows, false);
            let rz = linear2(&rz_product, 1.0, &r, a);
            let next_r_product = matmul_f16(&z, &rz, rows, rows, rows, false);
            r = linear2(&next_r_product, 1.0, &rz, a);
        }
    }

    matmul_f16(
        q.as_ref().expect("final Q is set"),
        &x,
        rows,
        cols,
        rows,
        false,
    )
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
