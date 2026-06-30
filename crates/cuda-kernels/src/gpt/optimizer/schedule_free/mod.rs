use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::amax::{amax4_f32, max4_f32};
use crate::block_reduce::block_max_store_f32;
use crate::float_ptx::abs_f32;
use crate::nvfp4_quant::kernels::convert::cvt_rn_satfinite_e2m1x2_f32;
use crate::nvfp4_quant::kernels::four_six::helpers::{
    GROUP_SIZE, four_six_block_group, four_six_global_scale, four_six_group_scale, four_six_lane,
};
use crate::warp_reduce::thread_lane_warp;

use super::threads::WARPS_PER_BLOCK;

const TENSOR_AMAX_VALUES_PER_BLOCK: u32 =
    crate::nvfp4_quant::kernels::row_amax::TENSOR_AMAX_VALUES_PER_BLOCK;
const SCALE_OVERRIDE: f32 = 1.0;

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
        let (tid, lane, warp_in_block) = thread_lane_warp();
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

        block_max_store_f32!(TENSOR_AMAX, out[chunk], local_amax, lane, warp_in_block);
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
        let (lane_in_group, group_mask, group_leader) = four_six_lane();
        let group = four_six_block_group();

        if group < out_scales.len() {
            let base = group * GROUP_SIZE;
            let tensor_amax = amax[0];
            let global_scale = four_six_global_scale(tensor_amax, SCALE_OVERRIDE);
            unsafe {
                if group == 0 && lane_in_group == 0 {
                    *out_global_scale.get_unchecked_mut(0) = global_scale;
                }

                let value = schedule_value(z_master, x_master, beta, (base + lane_in_group) as u32);
                let (scale_bits, inv_scale) = four_six_group_scale(
                    value,
                    global_scale,
                    SCALE_OVERRIDE,
                    group_mask,
                    group_leader,
                    lane_in_group,
                );

                if lane_in_group == 0 {
                    *out_scales.get_unchecked_mut(group) = scale_bits;
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
