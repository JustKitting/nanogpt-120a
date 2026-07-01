use crate::polar_reference::{normalized_polar_source, polar_iterations_f16};

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
    let update = polar_iterations_f16(polar, polar_rows, polar_cols, iterations);
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
