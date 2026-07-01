const NORM_SAFETY: f32 = 1.01;
const NORM_EPS: f32 = 1.0e-7;

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
