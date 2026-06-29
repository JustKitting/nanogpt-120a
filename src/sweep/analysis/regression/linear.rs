use super::super::stats::{EPS, dot};

const RIDGE_LAMBDA: f64 = 1.0;

pub fn ridge_fit(x: &[Vec<f64>], y: &[f64]) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    let p = x.first()?.len();
    let mut xtx = vec![vec![0.0; p]; p];
    let mut xty = vec![0.0; p];
    for (row, y) in x.iter().zip(y) {
        for i in 0..p {
            xty[i] += row[i] * y;
            for j in 0..p {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    for (i, row) in xtx.iter_mut().enumerate() {
        row[i] += RIDGE_LAMBDA;
    }
    let inverse = invert(xtx)?;
    let beta = inverse.iter().map(|row| dot(row, &xty)).collect();
    Some((beta, inverse))
}

pub fn residual_std(x: &[Vec<f64>], y: &[f64], beta: &[f64]) -> f64 {
    let rss = x
        .iter()
        .zip(y)
        .map(|(row, y)| {
            let err = y - dot(row, beta);
            err * err
        })
        .sum::<f64>();
    (rss / x.len().saturating_sub(1).max(1) as f64).sqrt()
}

pub fn quadratic_form(x: &[f64], a: &[Vec<f64>]) -> f64 {
    let ax = a.iter().map(|row| dot(row, x)).collect::<Vec<_>>();
    dot(x, &ax)
}

fn invert(mut a: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut inv = vec![vec![0.0; n]; n];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for col in 0..n {
        let pivot =
            (col..n).max_by(|&a_i, &b_i| a[a_i][col].abs().total_cmp(&a[b_i][col].abs()))?;
        if a[pivot][col].abs() <= EPS {
            return None;
        }
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let scale = a[col][col];
        for j in 0..n {
            a[col][j] /= scale;
            inv[col][j] /= scale;
        }
        eliminate_column(&mut a, &mut inv, col);
    }
    Some(inv)
}

fn eliminate_column(a: &mut [Vec<f64>], inv: &mut [Vec<f64>], col: usize) {
    for row in 0..a.len() {
        if row == col {
            continue;
        }
        let factor = a[row][col];
        for j in 0..a.len() {
            a[row][j] -= factor * a[col][j];
            inv[row][j] -= factor * inv[col][j];
        }
    }
}
