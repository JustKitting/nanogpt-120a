use crate::polar_reference::{matmul_f16, normalized_polar_source, polar_next};

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
        let gram = matmul_f16(&source, &source, rows, rows, cols, true);
        let ax = matmul_f16(&gram, &source, rows, cols, rows, false);
        let aax = matmul_f16(&gram, &ax, rows, cols, rows, false);
        source = polar_next(&source, &ax, &aax, iter);
    }
    source
}
