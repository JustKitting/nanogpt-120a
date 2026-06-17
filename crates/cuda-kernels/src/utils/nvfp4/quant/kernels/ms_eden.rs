use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::float_ptx::abs_f32;
use crate::nvfp4_cast::{e2m1_value, e4m3_value};
use crate::warp_reduce::{half_warp_max_f32, half_warp_sum_f32, warp_max_f32};

use super::convert::{cvt_rn_satfinite_e2m1x2_f32, cvt_rn_satfinite_e4m3x2_f32};

#[allow(static_mut_refs)]
#[cuda_module]
pub(crate) mod module {
    use super::*;

    const HADAMARD_DIM: u32 = 32;
    const GROUP_SIZE: u32 = 16;
    const INV_SQRT_32: f32 = 0.176_776_69;
    const FP4_MAX: f32 = 6.0;

    static mut ROTATED: SharedArray<f32, { HADAMARD_DIM as usize }> = SharedArray::UNINIT;

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn fp32_to_nvfp4_ms_eden_kernel(
        x: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        mut out_chunk_amax: DisjointSlice<f32>,
        row_len: u32,
        global_scale: f32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let lane = warp::lane_id();
        let chunk = thread::blockIdx_x();
        let chunk_base = chunk * HADAMARD_DIM;
        let element = chunk_base + lane;
        let row = element / row_len;
        let row_offset = element - row * row_len;

        let value = hadamard_value(x, chunk_base, lane, sign_seed);
        unsafe {
            ROTATED[lane as usize] = value;
        }
        thread::sync_threads();

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

        let lane_in_group = lane & 0x0f;
        let group_mask = if lane < GROUP_SIZE {
            0x0000_ffff
        } else {
            0xffff_0000
        };
        let group_leader = lane & !0x0f;
        let group = chunk * 2 + lane / GROUP_SIZE;
        let safe_global_scale = if global_scale == 0.0 {
            1.0
        } else {
            global_scale
        };
        let group_amax = half_warp_max_f32(abs_f32(value), group_mask);
        let scale_bits = cvt_rn_satfinite_e4m3x2_f32(
            0.0,
            group_amax * scale_override / (FP4_MAX * safe_global_scale),
        );
        let scale = nonzero_scale(e4m3_value(scale_bits as u16));
        let x_scaled = value / (scale * safe_global_scale);
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

        thread::sync_threads();

        if lane_in_group < GROUP_SIZE / 2 {
            let pair = group_leader + lane_in_group * 2;
            let byte = chunk_base / 2 + (lane / GROUP_SIZE) * (GROUP_SIZE / 2) + lane_in_group;
            let hi = unsafe { ROTATED[pair as usize] } / (scale * safe_global_scale);
            let lo = unsafe { ROTATED[pair as usize + 1] } / (scale * safe_global_scale);
            unsafe {
                *out_fp4.get_unchecked_mut(byte as usize) = cvt_rn_satfinite_e2m1x2_f32(hi, lo);
            }
        }
    }

    #[inline(always)]
    fn hadamard_value(x: &[f32], chunk_base: u32, lane: u32, seed: u32) -> f32 {
        let mut sum = 0.0_f32;
        let mut col = 0;
        while col < HADAMARD_DIM {
            let input = x[(chunk_base + col) as usize];
            let sign = hadamard_sign(col, lane) * random_sign(seed, chunk_base + lane);
            sum += input * sign;
            col += 1;
        }
        sum * INV_SQRT_32
    }

    #[inline(always)]
    fn hadamard_sign(row: u32, col: u32) -> f32 {
        if parity(row & col) == 0 { 1.0 } else { -1.0 }
    }

    #[inline(always)]
    fn parity(mut value: u32) -> u32 {
        value ^= value >> 16;
        value ^= value >> 8;
        value ^= value >> 4;
        value ^= value >> 2;
        value ^= value >> 1;
        value & 1
    }

    #[inline(always)]
    fn random_sign(seed: u32, index: u32) -> f32 {
        if hash_u32(seed ^ index) & 1 == 0 {
            1.0
        } else {
            -1.0
        }
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

    #[inline(always)]
    fn random_unit_f32(seed: u32, index: u32) -> f32 {
        let bits = hash_u32(seed ^ index) & 0x00ff_ffff;
        bits as f32 * 5.960_464_5e-8
    }

    #[inline(always)]
    fn hash_u32(mut value: u32) -> u32 {
        value ^= value >> 16;
        value = value.wrapping_mul(0x7feb_352d);
        value ^= value >> 15;
        value = value.wrapping_mul(0x846c_a68b);
        value ^ (value >> 16)
    }

    #[inline(always)]
    fn nonzero_scale(scale: f32) -> f32 {
        if scale == 0.0 { 1.0 } else { scale }
    }
}
