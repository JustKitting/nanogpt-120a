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
    let ax = cols as f32 * xh * gramh;
    let axh = round_f16_to_f32(ax);
    let aax = cols as f32 * axh * gramh;
    let base = (8.287_212 / 1.01_f32).mul_add(x, (-23.595_886 / 1.030_301_f32) * ax);
    (17.300_388 / 1.051_010_1_f32).mul_add(aax, base)
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
