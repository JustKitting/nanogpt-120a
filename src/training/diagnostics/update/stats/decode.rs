pub(super) fn nvfp4_host_value(
    bytes: &[u8],
    scales: &[u8],
    global_scale: f32,
    index: usize,
) -> f32 {
    let byte = bytes[index / 2];
    let payload = if index & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    };
    e2m1_host_value(payload) * e4m3_host_value(scales[index / 16]) * global_scale
}

fn e2m1_host_value(bits: u8) -> f32 {
    const VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let value = VALUES[(bits & 0x7) as usize];
    if bits & 0x8 == 0 { value } else { -value }
}

fn e4m3_host_value(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0 {
        sign * (mantissa as f32) * 2.0_f32.powi(-9)
    } else {
        sign * (1.0 + mantissa as f32 / 8.0) * 2.0_f32.powi(exponent as i32 - 7)
    }
}
