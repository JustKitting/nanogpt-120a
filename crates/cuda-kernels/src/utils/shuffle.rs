use cuda_device::warp;

#[inline(always)]
pub fn xor_f32_sync(mask: u32, value: f32, lane_mask: u32) -> f32 {
    warp::shuffle_xor_f32_sync(mask, value, lane_mask)
}
