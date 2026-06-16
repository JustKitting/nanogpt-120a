#[path = "inline_ptx.rs"]
pub mod inline_ptx;
#[path = "shuffle.rs"]
pub mod shuffle;

pub use inline_ptx::{abs_f32, e2m1_value, e4m3_value, fma_f32, max_f32, sqrt_f32};

pub const FULL_WARP_MASK: u32 = 0xffff_ffff;

#[inline(always)]
pub fn warp_sum_f32(mut value: f32) -> f32 {
    value += shuffle::xor_f32_sync(FULL_WARP_MASK, value, 16);
    value += shuffle::xor_f32_sync(FULL_WARP_MASK, value, 8);
    value += shuffle::xor_f32_sync(FULL_WARP_MASK, value, 4);
    value += shuffle::xor_f32_sync(FULL_WARP_MASK, value, 2);
    value + shuffle::xor_f32_sync(FULL_WARP_MASK, value, 1)
}
