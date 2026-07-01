#![allow(dead_code)]

pub fn tc_f16(value: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(value))
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;

    if exp == 0xff {
        return sign | if mant == 0 { 0x7c00 } else { 0x7e00 };
    }

    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x80_0000;
        let shift = (14 - half_exp) as u32;
        let mut half_mant = (mantissa >> shift) as u16;
        let round = (mantissa >> (shift - 1)) & 1;
        let sticky = mantissa & ((1_u32 << (shift - 1)) - 1);
        if round != 0 && (sticky != 0 || (half_mant & 1) != 0) {
            half_mant += 1;
        }
        return sign | half_mant;
    }

    let mut half = sign | ((half_exp as u16) << 10) | ((mant >> 13) as u16);
    let round = (mant >> 12) & 1;
    let sticky = mant & 0x0fff;
    if round != 0 && (sticky != 0 || (half & 1) != 0) {
        half += 1;
    }
    half
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let mant = (bits & 0x03ff) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut mantissa = mant;
        let mut exponent = -14;
        while mantissa & 0x0400 == 0 {
            mantissa <<= 1;
            exponent -= 1;
        }
        mantissa &= 0x03ff;
        let f_exp = ((exponent + 127) as u32) << 23;
        return f32::from_bits(sign | f_exp | (mantissa << 13));
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }

    let f_exp = ((exp - 15 + 127) as u32) << 23;
    f32::from_bits(sign | f_exp | (mant << 13))
}
