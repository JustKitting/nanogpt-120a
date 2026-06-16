use cuda_device::warp;

pub const FULL_WARP_MASK: u32 = 0xffff_ffff;

#[inline(always)]
pub fn warp_sum_f32(mut value: f32) -> f32 {
    value += warp::shuffle_xor_f32_sync(FULL_WARP_MASK, value, 16);
    value += warp::shuffle_xor_f32_sync(FULL_WARP_MASK, value, 8);
    value += warp::shuffle_xor_f32_sync(FULL_WARP_MASK, value, 4);
    value += warp::shuffle_xor_f32_sync(FULL_WARP_MASK, value, 2);
    value + warp::shuffle_xor_f32_sync(FULL_WARP_MASK, value, 1)
}
