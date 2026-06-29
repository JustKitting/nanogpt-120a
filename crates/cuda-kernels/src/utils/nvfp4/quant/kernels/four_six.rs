use cuda_device::{DisjointSlice, cuda_module, kernel, thread, warp};

use crate::float_ptx::abs_f32;
use crate::warp_reduce::{half_warp_max_f32, half_warp_sum_f32};

use super::convert::{
    candidate_error, cvt_rn_satfinite_e2m1x2_f32, local_scale_bits, nonzero_global_scale,
    nvfp4_inv_scale, scale_value,
};

#[cuda_module]
pub(crate) mod module {
    use super::*;

    const GROUP_SIZE: usize = 16;
    const FP4_MAX: f32 = 6.0;
    const FP8_MAX_FOUR_SIX: f32 = 256.0;

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_kernel(
        x: &[f32],
        amax: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scale: DisjointSlice<f32>,
        row_len: u32,
        scale_override: f32,
    ) {
        let lane = warp::lane_id() as usize;
        let lane_in_group = lane & 0x0f;
        let group_mask = if lane < GROUP_SIZE {
            0x0000_ffff
        } else {
            0xffff_0000
        };
        let group_leader = (lane & !0x0f) as u32;
        let groups_per_block = thread::blockDim_x() as usize / GROUP_SIZE;
        let group = thread::blockIdx_x() as usize * groups_per_block
            + thread::threadIdx_x() as usize / GROUP_SIZE;

        if group < out_scales.len() {
            let base = group * GROUP_SIZE;
            let row_len = row_len as usize;
            let scalar_scale = row_len == 0;
            let scale_row_len = if scalar_scale { usize::MAX } else { row_len };
            let row = base / scale_row_len;
            let global_scale = four_six_global_scale(amax[row], scale_override);
            let writes_global_scale = if scalar_scale {
                group == 0
            } else {
                base == row * scale_row_len
            };

            unsafe {
                if writes_global_scale && lane_in_group == 0 {
                    *out_global_scale.get_unchecked_mut(row) = global_scale;
                }

                let value = x[base + lane_in_group];
                let (scale_bits, inv_scale) = four_six_group_scale(
                    value,
                    global_scale,
                    scale_override,
                    group_mask,
                    group_leader,
                    lane_in_group,
                );

                if lane_in_group == 0 {
                    *out_scales.get_unchecked_mut(group) = scale_bits;
                }

                if lane_in_group < GROUP_SIZE / 2 {
                    let pair = lane_in_group * 2;
                    let hi = x[base + pair] * inv_scale;
                    let lo = x[base + pair + 1] * inv_scale;
                    *out_fp4.get_unchecked_mut(base / 2 + lane_in_group) =
                        cvt_rn_satfinite_e2m1x2_f32(lo, hi);
                }
            }
        }
    }

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_rowwise_pow2_kernel(
        x: &[f32],
        amax: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scale: DisjointSlice<f32>,
        row_shift: u32,
        row_mask: u32,
        scale_override: f32,
    ) {
        let lane = warp::lane_id() as usize;
        let lane_in_group = lane & 0x0f;
        let group_mask = if lane < GROUP_SIZE {
            0x0000_ffff
        } else {
            0xffff_0000
        };
        let group_leader = (lane & !0x0f) as u32;
        let groups_per_block = thread::blockDim_x() as usize / GROUP_SIZE;
        let group = thread::blockIdx_x() as usize * groups_per_block
            + thread::threadIdx_x() as usize / GROUP_SIZE;

        let base = group * GROUP_SIZE;
        let row = (base as u32 >> row_shift) as usize;
        let global_scale = four_six_global_scale(amax[row], scale_override);

        unsafe {
            if (base as u32 & row_mask) == 0 && lane_in_group == 0 {
                *out_global_scale.get_unchecked_mut(row) = global_scale;
            }

            let value = x[base + lane_in_group];
            let (scale_bits, inv_scale) = four_six_group_scale(
                value,
                global_scale,
                scale_override,
                group_mask,
                group_leader,
                lane_in_group,
            );

            if lane_in_group == 0 {
                *out_scales.get_unchecked_mut(group) = scale_bits;
            }

            if lane_in_group < GROUP_SIZE / 2 {
                let pair = lane_in_group * 2;
                let hi = x[base + pair] * inv_scale;
                let lo = x[base + pair + 1] * inv_scale;
                *out_fp4.get_unchecked_mut(base / 2 + lane_in_group) =
                    cvt_rn_satfinite_e2m1x2_f32(lo, hi);
            }
        }
    }

    #[inline(always)]
    fn four_six_global_scale(tensor_amax: f32, scale_override: f32) -> f32 {
        nonzero_global_scale(if tensor_amax == 0.0 {
            1.0
        } else {
            tensor_amax * scale_override / (FP8_MAX_FOUR_SIX * FP4_MAX)
        })
    }

    #[inline(always)]
    fn four_six_group_scale(
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
        scale_bits_four =
            warp::shuffle_sync(group_mask, scale_bits_four as u32, group_leader) as u16;
        scale_six = warp::shuffle_f32_sync(group_mask, scale_six, group_leader);
        scale_four = warp::shuffle_f32_sync(group_mask, scale_four, group_leader);

        let err_six =
            half_warp_sum_f32(candidate_error(value, scale_six, global_scale), group_mask);
        let err_four =
            half_warp_sum_f32(candidate_error(value, scale_four, global_scale), group_mask);
        let grid_max = if err_six <= err_four { 6.0 } else { 4.0 };
        let scale_bits = if grid_max == 6.0 {
            scale_bits_six
        } else {
            scale_bits_four
        };
        let scale = if grid_max == 6.0 {
            scale_six
        } else {
            scale_four
        };
        (scale_bits as u8, nvfp4_inv_scale(scale, global_scale))
    }
}
