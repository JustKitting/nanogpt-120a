use cuda_device::{DisjointSlice, thread, warp};

use crate::float_ptx::abs_f32;
use crate::nvfp4_cast::{e2m1_value, e4m3_value};
use crate::warp_reduce::{half_warp_max_f32, half_warp_sum_f32, warp_max_f32};

use super::super::convert::{
    cvt_rn_satfinite_e2m1x2_f32, cvt_rn_satfinite_e4m3x2_f32, nonzero_global_scale, nonzero_scale,
    nvfp4_inv_scale,
};
use super::random::random_unit_f32;
use super::{AMAX_WARPS_PER_BLOCK, FP4_MAX, GROUP_SIZE, HADAMARD_DIM, INV_SQRT_32};

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub(super) fn ms_eden_pack_chunk(
    input: f32,
    out_fp4: &mut DisjointSlice<'_, u8>,
    out_scales: &mut DisjointSlice<'_, u8>,
    out_global_scales: &mut DisjointSlice<'_, f32>,
    out_chunk_amax: &mut DisjointSlice<'_, f32>,
    chunk: u32,
    dst_row_len: u32,
    global_scale: f32,
    scale_override: f32,
    scale_seed: u32,
) {
    let lane = warp::lane_id();
    let chunk_base = chunk * HADAMARD_DIM;
    let element = chunk_base + lane;
    let row = element / dst_row_len;
    let row_offset = element - row * dst_row_len;

    let value = hadamard_transform_lane(input, lane);

    if row_offset == 0 {
        unsafe {
            *out_global_scales.get_unchecked_mut(row as usize) = global_scale;
        }
    }

    let chunk_amax = warp_max_f32(abs_f32(value));
    if lane == 0 {
        unsafe {
            *out_chunk_amax.get_unchecked_mut(chunk as usize) = chunk_amax;
        }
    }

    ms_eden_pack_payload(
        value,
        out_fp4,
        out_scales,
        chunk,
        global_scale,
        scale_override,
        scale_seed,
    );
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub(super) fn ms_eden_pack_chunk_no_chunk_amax(
    input: f32,
    out_fp4: &mut DisjointSlice<'_, u8>,
    out_scales: &mut DisjointSlice<'_, u8>,
    out_global_scales: &mut DisjointSlice<'_, f32>,
    chunk: u32,
    dst_row_len: u32,
    global_scale: f32,
    scale_override: f32,
    scale_seed: u32,
) {
    let lane = warp::lane_id();
    let chunk_base = chunk * HADAMARD_DIM;
    let element = chunk_base + lane;
    let row = element / dst_row_len;
    let row_offset = element - row * dst_row_len;

    let value = hadamard_transform_lane(input, lane);

    if row_offset == 0 {
        unsafe {
            *out_global_scales.get_unchecked_mut(row as usize) = global_scale;
        }
    }

    ms_eden_pack_payload(
        value,
        out_fp4,
        out_scales,
        chunk,
        global_scale,
        scale_override,
        scale_seed,
    );
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub(super) fn ms_eden_pack_chunk_no_chunk_amax_row(
    input: f32,
    out_fp4: &mut DisjointSlice<'_, u8>,
    out_scales: &mut DisjointSlice<'_, u8>,
    out_global_scales: &mut DisjointSlice<'_, f32>,
    chunk: u32,
    row: u32,
    first_chunk_in_row: bool,
    global_scale: f32,
    scale_override: f32,
    scale_seed: u32,
) {
    let lane = warp::lane_id();
    let value = hadamard_transform_lane(input, lane);

    if first_chunk_in_row && lane == 0 {
        unsafe {
            *out_global_scales.get_unchecked_mut(row as usize) = global_scale;
        }
    }

    ms_eden_pack_payload(
        value,
        out_fp4,
        out_scales,
        chunk,
        global_scale,
        scale_override,
        scale_seed,
    );
}

#[inline(always)]
fn ms_eden_pack_payload(
    value: f32,
    out_fp4: &mut DisjointSlice<'_, u8>,
    out_scales: &mut DisjointSlice<'_, u8>,
    chunk: u32,
    global_scale: f32,
    scale_override: f32,
    scale_seed: u32,
) {
    let lane = warp::lane_id();
    let chunk_base = chunk * HADAMARD_DIM;
    let lane_in_group = lane & 0x0f;
    let group_mask = if lane < GROUP_SIZE {
        0x0000_ffff
    } else {
        0xffff_0000
    };
    let group_leader = lane & !0x0f;
    let group = chunk * 2 + lane / GROUP_SIZE;
    let safe_global_scale = nonzero_global_scale(global_scale);
    let group_amax = half_warp_max_f32(abs_f32(value), group_mask);
    let scale_bits = cvt_rn_satfinite_e4m3x2_f32(
        0.0,
        group_amax * scale_override * nvfp4_inv_scale(FP4_MAX, safe_global_scale),
    );
    let scale = nonzero_scale(e4m3_value(scale_bits as u16));
    let inv_scale = nvfp4_inv_scale(scale, safe_global_scale);
    let x_scaled = value * inv_scale;
    let payload = cvt_rn_satfinite_e2m1x2_f32(0.0, x_scaled) & 0x0f;
    let fp4_value = e2m1_value(payload);

    let num = half_warp_sum_f32(x_scaled * x_scaled, group_mask);
    let denom = half_warp_sum_f32(x_scaled * fp4_value, group_mask);
    let correction = if denom == 0.0 { 1.0 } else { num / denom };
    let corrected_scale = nonzero_scale(scale * correction);
    let rounded_scale_bits = stochastic_e4m3_scale(corrected_scale, scale_seed, group);

    if lane == group_leader {
        unsafe {
            *out_scales.get_unchecked_mut(group as usize) = rounded_scale_bits;
        }
    }

    let pair = group_leader + (lane_in_group & 0x7) * 2;
    let hi_value = warp::shuffle_f32_sync(0xffff_ffff, value, pair);
    let lo_value = warp::shuffle_f32_sync(0xffff_ffff, value, pair + 1);
    if lane_in_group < GROUP_SIZE / 2 {
        let byte = chunk_base / 2 + (lane / GROUP_SIZE) * (GROUP_SIZE / 2) + lane_in_group;
        let hi = hi_value * inv_scale;
        let lo = lo_value * inv_scale;
        unsafe {
            *out_fp4.get_unchecked_mut(byte as usize) = cvt_rn_satfinite_e2m1x2_f32(lo, hi);
        }
    }
}

#[inline(always)]
pub(super) fn pack_chunk() -> u32 {
    thread::blockIdx_x() * AMAX_WARPS_PER_BLOCK + thread::threadIdx_x() / 32
}

#[inline(always)]
fn hadamard_transform_lane(mut value: f32, lane: u32) -> f32 {
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

#[inline(always)]
fn stochastic_e4m3_scale(value: f32, seed: u32, group: u32) -> u8 {
    let curr_bits = cvt_rn_satfinite_e4m3x2_f32(0.0, value);
    let curr = e4m3_value(curr_bits as u16);
    let prev_bits = curr_bits.saturating_sub(1);
    let next_bits = curr_bits.saturating_add(1);
    let prev = e4m3_value(prev_bits as u16);
    let next = e4m3_value(next_bits as u16);
    let up = if curr > value { curr } else { next };
    let down = if curr > value { prev } else { curr };
    let up_bits = if curr > value { curr_bits } else { next_bits };
    let down_bits = if curr > value { prev_bits } else { curr_bits };
    let denom = up - down;
    let prob_up = if denom == 0.0 {
        0.0
    } else {
        (value - down) / denom
    };

    if random_unit_f32(seed, group) < prob_up {
        up_bits
    } else {
        down_bits
    }
}
