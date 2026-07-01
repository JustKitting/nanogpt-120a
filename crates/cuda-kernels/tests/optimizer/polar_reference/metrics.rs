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

pub fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}
