use crate::polar_coefficients::coefficients;
use crate::polar_reference::{matmul_f16, polar_next};

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
