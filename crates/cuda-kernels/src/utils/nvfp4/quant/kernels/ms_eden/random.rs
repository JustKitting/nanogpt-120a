#[inline(always)]
pub(super) fn random_sign(seed: u32, index: u32) -> f32 {
    if hash_u32(seed ^ index) & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

#[inline(always)]
pub(super) fn random_unit_f32(seed: u32, index: u32) -> f32 {
    let bits = hash_u32(seed ^ index) & 0x00ff_ffff;
    bits as f32 * 5.960_464_5e-8
}

#[inline(always)]
fn hash_u32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}
