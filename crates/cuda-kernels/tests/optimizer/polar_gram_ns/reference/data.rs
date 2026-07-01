pub fn gradient(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.0009 + ((i / cols) as f32) * 0.00002)
        .collect()
}
