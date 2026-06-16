use crate::kernel_ops::{e2m1_value, e4m3_value};

#[inline(always)]
pub fn nvfp4_value(bytes: &[u8], scales: &[u8], global_scale: f32, index: usize) -> f32 {
    let byte = bytes[index / 2];
    let payload = if index & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    };

    e2m1_value(payload) * e4m3_value(scales[index / 16] as u16) * global_scale
}
