use crate::float_ptx::max_f32;
use crate::shuffle;

const FULL_WARP_MASK: u32 = 0xffff_ffff;

#[inline(always)]
pub fn warp_sum_f32(mut value: f32) -> f32 {
    value += shuffle::xor_f32_sync(FULL_WARP_MASK, value, 16);
    value += shuffle::xor_f32_sync(FULL_WARP_MASK, value, 8);
    value += shuffle::xor_f32_sync(FULL_WARP_MASK, value, 4);
    value += shuffle::xor_f32_sync(FULL_WARP_MASK, value, 2);
    value + shuffle::xor_f32_sync(FULL_WARP_MASK, value, 1)
}

#[inline(always)]
pub fn warp_max_f32(mut value: f32) -> f32 {
    value = max_f32(value, shuffle::xor_f32_sync(FULL_WARP_MASK, value, 16));
    value = max_f32(value, shuffle::xor_f32_sync(FULL_WARP_MASK, value, 8));
    value = max_f32(value, shuffle::xor_f32_sync(FULL_WARP_MASK, value, 4));
    value = max_f32(value, shuffle::xor_f32_sync(FULL_WARP_MASK, value, 2));
    max_f32(value, shuffle::xor_f32_sync(FULL_WARP_MASK, value, 1))
}

#[inline(always)]
pub fn half_warp_sum_f32(mut value: f32, mask: u32) -> f32 {
    value += shuffle::xor_f32_sync(mask, value, 8);
    value += shuffle::xor_f32_sync(mask, value, 4);
    value += shuffle::xor_f32_sync(mask, value, 2);
    value + shuffle::xor_f32_sync(mask, value, 1)
}

#[inline(always)]
pub fn half_warp_max_f32(mut value: f32, mask: u32) -> f32 {
    value = max_f32(value, shuffle::xor_f32_sync(mask, value, 8));
    value = max_f32(value, shuffle::xor_f32_sync(mask, value, 4));
    value = max_f32(value, shuffle::xor_f32_sync(mask, value, 2));
    max_f32(value, shuffle::xor_f32_sync(mask, value, 1))
}
