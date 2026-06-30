use cuda_device::warp;

use crate::float_ptx::abs_f32;
use crate::warp_reduce::{half_warp_max_f32, half_warp_sum_f32};

use super::super::convert::{
    candidate_error, local_scale_bits, nonzero_global_scale, nvfp4_inv_scale, scale_value,
};

pub(super) const GROUP_SIZE: usize = 16;

const FP4_MAX: f32 = 6.0;
const FP8_MAX_FOUR_SIX: f32 = 256.0;

#[inline(always)]
pub(super) fn four_six_global_scale(tensor_amax: f32, scale_override: f32) -> f32 {
    nonzero_global_scale(if tensor_amax == 0.0 {
        1.0
    } else {
        tensor_amax * scale_override / (FP8_MAX_FOUR_SIX * FP4_MAX)
    })
}

#[inline(always)]
pub(super) fn four_six_group_scale(
    value: f32,
    global_scale: f32,
    scale_override: f32,
    group_mask: u32,
    group_leader: u32,
    lane_in_group: usize,
) -> (u8, f32) {
    let group_amax = half_warp_max_f32(abs_f32(value), group_mask);
    let mut scale_bits_six = 0u16;
    let mut scale_bits_four = 0u16;
    let mut scale_six = 0.0;
    let mut scale_four = 0.0;

    if lane_in_group == 0 {
        scale_bits_six = local_scale_bits(group_amax, global_scale, scale_override, 6.0);
        scale_bits_four = local_scale_bits(group_amax, global_scale, scale_override, 4.0);
        scale_six = scale_value(scale_bits_six);
        scale_four = scale_value(scale_bits_four);
    }

    scale_bits_six = warp::shuffle_sync(group_mask, scale_bits_six as u32, group_leader) as u16;
    scale_bits_four = warp::shuffle_sync(group_mask, scale_bits_four as u32, group_leader) as u16;
    scale_six = warp::shuffle_f32_sync(group_mask, scale_six, group_leader);
    scale_four = warp::shuffle_f32_sync(group_mask, scale_four, group_leader);

    let err_six = half_warp_sum_f32(candidate_error(value, scale_six, global_scale), group_mask);
    let err_four = half_warp_sum_f32(candidate_error(value, scale_four, global_scale), group_mask);
    let (scale_bits, scale) = if err_six <= err_four {
        (scale_bits_six, scale_six)
    } else {
        (scale_bits_four, scale_four)
    };
    (scale_bits as u8, nvfp4_inv_scale(scale, global_scale))
}
