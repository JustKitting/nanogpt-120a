use super::round_f16_to_f32;

pub fn polar_first_iteration_scalar(x: f32, rows: usize, cols: usize) -> f32 {
    let (rows, cols) = if rows >= cols {
        (rows, cols)
    } else {
        (cols, rows)
    };
    let inv_norm = 1.0 / (((rows * cols) as f32 * x * x).sqrt() * 1.01 + 1.0e-7);
    let x = x * inv_norm;
    let xh = round_f16_to_f32(x);
    let gram = rows as f32 * xh * xh;
    let gramh = round_f16_to_f32(gram);
    let gram2 = cols as f32 * gramh * gramh;
    let off_diag =
        (17.300_388 / 1.051_010_1_f32).mul_add(gram2, (-23.595_886 / 1.030_301_f32) * gram);
    let diag = off_diag + 8.287_212 / 1.01_f32;
    xh * (round_f16_to_f32(diag) + (cols as f32 - 1.0) * round_f16_to_f32(off_diag))
}
