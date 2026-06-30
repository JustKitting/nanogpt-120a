use cuda_device::{DisjointSlice, cuda_module, kernel, thread, warp};

use super::convert::cvt_rn_satfinite_e2m1x2_f32;

#[path = "four_six/helpers.rs"]
pub(crate) mod helpers;

#[cuda_module]
pub(crate) mod module {
    use super::helpers::*;
    use super::*;

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
}
