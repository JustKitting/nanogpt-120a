use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::amax::{amax4_f32, max4_f32};
use crate::float_ptx::abs_f32;
use crate::nvfp4_quant::kernels::convert::{
    candidate_error, cvt_rn_satfinite_e2m1x2_f32, local_scale_bits, scale_value,
};
use crate::warp_reduce::{half_warp_max_f32, half_warp_sum_f32, warp_max_f32};

use super::threads::WARPS_PER_BLOCK;

const GROUP_SIZE: usize = 16;
const TENSOR_AMAX_VALUES_PER_BLOCK: u32 =
    crate::nvfp4_quant::kernels::row_amax::TENSOR_AMAX_VALUES_PER_BLOCK;
const FP4_MAX: f32 = 6.0;
const FP8_MAX_FOUR_SIX: f32 = 256.0;
const SCALE_OVERRIDE: f32 = 1.0;

#[allow(static_mut_refs)]
#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn schedule_free_chunk_amax_kernel(
        z_master: &[f32],
        x_master: &[f32],
        mut out: DisjointSlice<f32>,
        beta: f32,
        len: u32,
    ) {
        static mut TENSOR_AMAX: SharedArray<f32, { WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;

        let chunk = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = tid / 32;
        let base = chunk * TENSOR_AMAX_VALUES_PER_BLOCK;
        let stride = thread::blockDim_x();
        let i0 = base + tid;
        let i1 = i0 + stride;
        let i2 = i1 + stride;
        let i3 = i2 + stride;

        let local_amax = if base + TENSOR_AMAX_VALUES_PER_BLOCK <= len {
            amax4_f32(
                schedule_value(z_master, x_master, beta, i0),
                schedule_value(z_master, x_master, beta, i1),
                schedule_value(z_master, x_master, beta, i2),
                schedule_value(z_master, x_master, beta, i3),
            )
        } else {
            max4_f32(
                checked_abs_schedule_value(z_master, x_master, beta, i0, len),
                checked_abs_schedule_value(z_master, x_master, beta, i1, len),
                checked_abs_schedule_value(z_master, x_master, beta, i2, len),
                checked_abs_schedule_value(z_master, x_master, beta, i3, len),
            )
        };

        let warp_amax = warp_max_f32(local_amax);
        if lane == 0 {
            unsafe {
                TENSOR_AMAX[warp_in_block as usize] = warp_amax;
            }
        }
        thread::sync_threads();

        if warp_in_block == 0 {
            let partial = if lane < WARPS_PER_BLOCK {
                unsafe { TENSOR_AMAX[lane as usize] }
            } else {
                0.0
            };
            let block_amax = warp_max_f32(partial);
            if lane == 0 {
                unsafe {
                    *out.get_unchecked_mut(chunk as usize) = block_amax;
                }
            }
        }
    }

    #[kernel]
    pub fn schedule_free_four_six_kernel(
        z_master: &[f32],
        x_master: &[f32],
        amax: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scale: DisjointSlice<f32>,
        beta: f32,
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
            let tensor_amax = amax[0];
            let global_scale = if tensor_amax == 0.0 {
                1.0
            } else {
                tensor_amax * SCALE_OVERRIDE / (FP8_MAX_FOUR_SIX * FP4_MAX)
            };
            unsafe {
                if group == 0 && lane_in_group == 0 {
                    *out_global_scale.get_unchecked_mut(0) = global_scale;
                }

                let value = schedule_value(z_master, x_master, beta, (base + lane_in_group) as u32);
                let group_amax = half_warp_max_f32(abs_f32(value), group_mask);
                let (scale_bits, scale) =
                    scale_for_group(group_amax, value, global_scale, group_mask, group_leader);
                let scale_for_payload = if scale == 0.0 { 1.0 } else { scale };
                let inv_scale = 1.0 / (scale_for_payload * global_scale);

                if lane_in_group == 0 {
                    *out_scales.get_unchecked_mut(group) = scale_bits as u8;
                }
                if lane_in_group < GROUP_SIZE / 2 {
                    let pair = lane_in_group * 2;
                    let hi =
                        schedule_value(z_master, x_master, beta, (base + pair) as u32) * inv_scale;
                    let lo = schedule_value(z_master, x_master, beta, (base + pair + 1) as u32)
                        * inv_scale;
                    *out_fp4.get_unchecked_mut(base / 2 + lane_in_group) =
                        cvt_rn_satfinite_e2m1x2_f32(lo, hi);
                }
            }
        }
    }

    #[inline(always)]
    fn scale_for_group(
        group_amax: f32,
        value: f32,
        global_scale: f32,
        group_mask: u32,
        group_leader: u32,
    ) -> (u16, f32) {
        let lane_in_group = warp::lane_id() & 0x0f;
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
        let err_six =
            half_warp_sum_f32(candidate_error(value, scale_six, global_scale), group_mask);
        let err_four =
            half_warp_sum_f32(candidate_error(value, scale_four, global_scale), group_mask);
        if err_six <= err_four {
            (bits_six, scale_six)
        } else {
            (bits_four, scale_four)
        }
    }

    #[inline(always)]
    fn schedule_value(z_master: &[f32], x_master: &[f32], beta: f32, index: u32) -> f32 {
        let i = index as usize;
        let z = z_master[i];
        let x = x_master[i];
        z + beta * (x - z)
    }

    #[inline(always)]
    fn checked_abs_schedule_value(
        z_master: &[f32],
        x_master: &[f32],
        beta: f32,
        index: u32,
        len: u32,
    ) -> f32 {
        if index < len {
            abs_f32(schedule_value(z_master, x_master, beta, index))
        } else {
            0.0
        }
    }
}
