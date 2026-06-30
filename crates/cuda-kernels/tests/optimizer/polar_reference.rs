use crate::polar_coefficients::coefficients;

const NORM_SAFETY: f32 = 1.01;
const NORM_EPS: f32 = 1.0e-7;

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

pub fn polar_next(x: &[f32], ax: &[f32], aax: &[f32], iter: usize) -> Vec<f32> {
    let (a, b, c) = coefficients(iter);
    x.iter()
        .zip(ax)
        .zip(aax)
        .map(|((x, ax), aax)| c.mul_add(*aax, a.mul_add(*x, b * *ax)))
        .collect()
}

pub fn matmul_f16(
    a: &[f32],
    b: &[f32],
    rows: usize,
    cols: usize,
    k_len: usize,
    rhs_t: bool,
) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0;
            for k in 0..k_len {
                let bv = if rhs_t {
                    b[col * k_len + k]
                } else {
                    b[k * cols + col]
                };
                sum += round_f16_to_f32(a[row * k_len + k]) * round_f16_to_f32(bv);
            }
            out[row * cols + col] = sum;
        }
    }
    out
}

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

pub fn round_f16_to_f32(value: f32) -> f32 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;

    if exp <= 0 {
        return 0.0;
    }
    if exp >= 0x1f {
        return f32::INFINITY;
    }

    let mut half_mant = mant >> 13;
    let round_bits = mant & 0x1fff;
    if round_bits > 0x1000 || (round_bits == 0x1000 && half_mant & 1 != 0) {
        half_mant += 1;
        if half_mant == 0x400 {
            half_mant = 0;
            exp += 1;
        }
    }

    let half = sign | ((exp as u32) << 10) | half_mant;
    let exp = ((half >> 10) & 0x1f) as i32;
    let mant = half & 0x3ff;
    if exp == 0 {
        return 0.0;
    }

    f32::from_bits(((half & 0x8000) << 16) | (((exp - 15 + 127) as u32) << 23) | (mant << 13))
}
