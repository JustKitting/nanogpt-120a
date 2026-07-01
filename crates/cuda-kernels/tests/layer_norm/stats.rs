pub(super) fn sample_rows(row_count: usize, row_len: usize, period: usize, center: f32, scale: f32, row_offset_scale: f32) -> Vec<f32> {
    (0..row_count * row_len).map(|i| {
        ((i % period) as f32 - center) * scale + (i / row_len) as f32 * row_offset_scale
    }).collect()
}

pub(super) fn reference_row_stats(
    x: &[f32],
    row_count: usize,
    row_len: usize,
    epsilon: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut mean = vec![0.0f32; row_count];
    let mut inv_std = vec![0.0f32; row_count];
    for row in 0..row_count {
        let base = row * row_len;
        mean[row] = x[base..base + row_len].iter().sum::<f32>() / row_len as f32;
        let variance = x[base..base + row_len]
            .iter()
            .map(|value| {
                let centered = value - mean[row];
                centered * centered
            })
            .sum::<f32>()
            / row_len as f32;
        inv_std[row] = 1.0 / (variance + epsilon).sqrt();
    }
    (mean, inv_std)
}
