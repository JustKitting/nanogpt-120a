use cuda_device::warp;

use super::super::{HADAMARD_DIM, INV_SQRT_32};

#[inline(always)]
pub(super) fn hadamard_transform_lane(mut value: f32, lane: u32) -> f32 {
    let mut stride = 1;
    while stride < HADAMARD_DIM {
        let peer = warp::shuffle_xor_f32_sync(0xffff_ffff, value, stride);
        value = if lane & stride == 0 {
            value + peer
        } else {
            peer - value
        };
        stride <<= 1;
    }

    value * INV_SQRT_32
}
