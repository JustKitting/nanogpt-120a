use cuda_device::{thread, warp};

use crate::f16_tc_matmul::cta_tile::CTA_THREADS;
use crate::float_ptx::abs_f32;
use crate::nvfp4_quant::kernels::convert::{
    candidate_error, cvt_rn_satfinite_e2m1x2_f32, local_scale_bits, nonzero_global_scale,
    nvfp4_inv_scale, scale_value,
};
use crate::warp_reduce::{half_warp_max_f32, half_warp_sum_f32};

use super::super::super::super::work_grid::WorkGrid;
use crate::device_ptr::read_f32;

const GROUP_SIZE: u32 = 16;
const SCALE_OVERRIDE: f32 = 1.0;

pub(super) fn encode_four_six(
    x: *const f32,
    out_fp4: *mut u8,
    out_scales: *mut u8,
    out_global_scale: *mut f32,
    len: u32,
    work: WorkGrid,
) {
    let groups_per_block = CTA_THREADS / GROUP_SIZE;
    let group_count = len / GROUP_SIZE;
    let group_stride = work.blocks() * groups_per_block;
    let mut group = work.block() * groups_per_block + thread::threadIdx_x() / GROUP_SIZE;
    let global_scale = nonzero_global_scale(unsafe { *out_global_scale });

    while group < group_count {
        encode_group(x, out_fp4, out_scales, global_scale, group);
        group += group_stride;
    }
}

fn encode_group(
    x: *const f32,
    out_fp4: *mut u8,
    out_scales: *mut u8,
    global_scale: f32,
    group: u32,
) {
    let lane = warp::lane_id();
    let lane_in_group = lane & 0x0f;
    let group_mask = if lane < GROUP_SIZE {
        0x0000_ffff
    } else {
        0xffff_0000
    };
    let group_leader = lane & !0x0f;
    let base = group * GROUP_SIZE;
    let value = read_f32(x, base + lane_in_group);
    let group_amax = half_warp_max_f32(abs_f32(value), group_mask);
    let (scale_bits, scale) = scale_for_group(
        group_amax,
        value,
        global_scale,
        group_mask,
        group_leader,
        lane_in_group,
    );
    let inv_scale = nvfp4_inv_scale(scale, global_scale);

    unsafe {
        if lane_in_group == 0 {
            *out_scales.add(group as usize) = scale_bits as u8;
        }
        if lane_in_group < GROUP_SIZE / 2 {
            let pair = lane_in_group * 2;
            let hi = read_f32(x, base + pair) * inv_scale;
            let lo = read_f32(x, base + pair + 1) * inv_scale;
            *out_fp4.add((base / 2 + lane_in_group) as usize) = cvt_rn_satfinite_e2m1x2_f32(lo, hi);
        }
    }
}

fn scale_for_group(
    group_amax: f32,
    value: f32,
    global_scale: f32,
    group_mask: u32,
    group_leader: u32,
    lane_in_group: u32,
) -> (u16, f32) {
    let mut bits_six = 0u16;
    let mut bits_four = 0u16;
    let mut scale_six = 0.0;
    let mut scale_four = 0.0;
    if lane_in_group == 0 {
        bits_six = local_scale_bits(group_amax, global_scale, SCALE_OVERRIDE, 6.0);
        bits_four = local_scale_bits(group_amax, global_scale, SCALE_OVERRIDE, 4.0);
        scale_six = scale_value(bits_six);
        scale_four = scale_value(bits_four);
    }
    bits_six = warp::shuffle_sync(group_mask, bits_six as u32, group_leader) as u16;
    bits_four = warp::shuffle_sync(group_mask, bits_four as u32, group_leader) as u16;
    scale_six = warp::shuffle_f32_sync(group_mask, scale_six, group_leader);
    scale_four = warp::shuffle_f32_sync(group_mask, scale_four, group_leader);
    let err_six = half_warp_sum_f32(candidate_error(value, scale_six, global_scale), group_mask);
    let err_four = half_warp_sum_f32(candidate_error(value, scale_four, global_scale), group_mask);
    if err_six <= err_four {
        (bits_six, scale_six)
    } else {
        (bits_four, scale_four)
    }
}
