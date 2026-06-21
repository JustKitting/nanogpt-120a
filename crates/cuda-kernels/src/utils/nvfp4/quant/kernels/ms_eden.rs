use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::amax::{amax4_f32, max4_f32};
use crate::float_ptx::{abs_f32, max_f32};
use crate::nvfp4::{nvfp4_rowwise_value, nvfp4_value};
use crate::nvfp4_cast::{e2m1_value, e4m3_value};
use crate::quartet::{
    QUARTET_MS_EDEN_FP4_MAX, QUARTET_MS_EDEN_FP8_MAX, QUARTET_MS_EDEN_SCALE_OVERRIDE,
};
use crate::warp_reduce::{half_warp_max_f32, half_warp_sum_f32, warp_max_f32};

use super::convert::{cvt_rn_satfinite_e2m1x2_f32, cvt_rn_satfinite_e4m3x2_f32};
use super::row_amax::TENSOR_AMAX_VALUES_PER_BLOCK;

#[allow(static_mut_refs)]
#[cuda_module]
pub(crate) mod module {
    use super::*;

    const HADAMARD_DIM: u32 = 32;
    const GROUP_SIZE: u32 = 16;
    const INV_SQRT_32: f32 = 0.176_776_69;
    const FP4_MAX: f32 = 6.0;
    const AMAX_WARPS_PER_BLOCK: u32 = crate::nvfp4_quant::config::WARPS_PER_BLOCK;

    static mut AMAX_REDUCE: SharedArray<f32, { AMAX_WARPS_PER_BLOCK as usize }> =
        SharedArray::UNINIT;

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn fp32_to_nvfp4_ms_eden_kernel(
        x: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        mut out_chunk_amax: DisjointSlice<f32>,
        chunk_count: u32,
        src_row_len: u32,
        dst_row_len: u32,
        global_scale: f32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let chunk = pack_chunk();
        if chunk >= chunk_count {
            return;
        }

        fp32_to_nvfp4_ms_eden_body(
            x,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            &mut out_chunk_amax,
            chunk,
            src_row_len,
            dst_row_len,
            global_scale,
            scale_override,
            sign_seed,
            scale_seed,
        );
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn fp32_to_nvfp4_ms_eden_device_scale_kernel(
        x: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        mut out_chunk_amax: DisjointSlice<f32>,
        global_scale: &[f32],
        chunk_count: u32,
        src_row_len: u32,
        dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let chunk = pack_chunk();
        if chunk >= chunk_count {
            return;
        }

        fp32_to_nvfp4_ms_eden_body(
            x,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            &mut out_chunk_amax,
            chunk,
            src_row_len,
            dst_row_len,
            global_scale[0],
            scale_override,
            sign_seed,
            scale_seed,
        );
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel(
        x: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        mut out_chunk_amax: DisjointSlice<f32>,
        global_scale: &[f32],
        chunk_count: u32,
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let chunk = pack_chunk();
        if chunk >= chunk_count {
            return;
        }

        fp32_transpose_to_nvfp4_ms_eden_body(
            x,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            &mut out_chunk_amax,
            chunk,
            source_rows,
            dst_row_len,
            source_cols,
            global_scale[0],
            scale_override,
            sign_seed,
            scale_seed,
        );
    }

    #[kernel]
    pub fn rowwise_nvfp4_chunk_amax_kernel(
        bytes: &[u8],
        scales: &[u8],
        global_scales: &[f32],
        mut out: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
    ) {
        let chunk = thread::blockIdx_x();
        let thread = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = thread / 32;
        let base = chunk * TENSOR_AMAX_VALUES_PER_BLOCK;
        let element_count = rows * cols;
        let stride = thread::blockDim_x();
        let i0 = base + thread;
        let i1 = i0 + stride;
        let i2 = i1 + stride;
        let i3 = i2 + stride;

        let local_amax = if base + TENSOR_AMAX_VALUES_PER_BLOCK <= element_count {
            amax4_f32(
                rowwise_value_at(bytes, scales, global_scales, cols, i0),
                rowwise_value_at(bytes, scales, global_scales, cols, i1),
                rowwise_value_at(bytes, scales, global_scales, cols, i2),
                rowwise_value_at(bytes, scales, global_scales, cols, i3),
            )
        } else {
            max4_f32(
                checked_rowwise_abs_value(bytes, scales, global_scales, cols, i0, element_count),
                checked_rowwise_abs_value(bytes, scales, global_scales, cols, i1, element_count),
                checked_rowwise_abs_value(bytes, scales, global_scales, cols, i2, element_count),
                checked_rowwise_abs_value(bytes, scales, global_scales, cols, i3, element_count),
            )
        };

        let warp_amax = warp_max_f32(local_amax);
        if lane == 0 {
            unsafe {
                AMAX_REDUCE[warp_in_block as usize] = warp_amax;
            }
        }

        thread::sync_threads();

        if warp_in_block == 0 {
            let partial = if lane < AMAX_WARPS_PER_BLOCK {
                unsafe { AMAX_REDUCE[lane as usize] }
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
    pub fn nvfp4_chunk_amax_kernel(
        bytes: &[u8],
        scales: &[u8],
        global_scale: &[f32],
        mut out: DisjointSlice<f32>,
        element_count: u32,
    ) {
        let chunk = thread::blockIdx_x();
        let thread = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = thread / 32;
        let base = chunk * TENSOR_AMAX_VALUES_PER_BLOCK;
        let stride = thread::blockDim_x();
        let i0 = base + thread;
        let i1 = i0 + stride;
        let i2 = i1 + stride;
        let i3 = i2 + stride;

        let local_amax = if base + TENSOR_AMAX_VALUES_PER_BLOCK <= element_count {
            amax4_f32(
                nvfp4_value_at(bytes, scales, global_scale, i0),
                nvfp4_value_at(bytes, scales, global_scale, i1),
                nvfp4_value_at(bytes, scales, global_scale, i2),
                nvfp4_value_at(bytes, scales, global_scale, i3),
            )
        } else {
            max4_f32(
                checked_nvfp4_abs_value(bytes, scales, global_scale, i0, element_count),
                checked_nvfp4_abs_value(bytes, scales, global_scale, i1, element_count),
                checked_nvfp4_abs_value(bytes, scales, global_scale, i2, element_count),
                checked_nvfp4_abs_value(bytes, scales, global_scale, i3, element_count),
            )
        };

        let warp_amax = warp_max_f32(local_amax);
        if lane == 0 {
            unsafe {
                AMAX_REDUCE[warp_in_block as usize] = warp_amax;
            }
        }

        thread::sync_threads();

        if warp_in_block == 0 {
            let partial = if lane < AMAX_WARPS_PER_BLOCK {
                unsafe { AMAX_REDUCE[lane as usize] }
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
    #[allow(clippy::too_many_arguments)]
    pub fn nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scale: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        mut out_chunk_amax: DisjointSlice<f32>,
        global_scale: &[f32],
        chunk_count: u32,
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let lane = warp::lane_id();
        let chunk = pack_chunk();
        if chunk >= chunk_count {
            return;
        }

        let chunk_base = chunk * HADAMARD_DIM;
        let input = nvfp4_transposed_hadamard_input(
            bytes,
            scales,
            source_global_scale,
            chunk_base,
            lane,
            source_rows,
            source_cols,
            dst_row_len,
            sign_seed,
        );
        ms_eden_pack_chunk(
            input,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            &mut out_chunk_amax,
            chunk,
            dst_row_len,
            global_scale[0],
            scale_override,
            scale_seed,
        );
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scales: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        mut out_chunk_amax: DisjointSlice<f32>,
        global_scale: &[f32],
        chunk_count: u32,
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let lane = warp::lane_id();
        let chunk = pack_chunk();
        if chunk >= chunk_count {
            return;
        }

        let chunk_base = chunk * HADAMARD_DIM;
        let input = rowwise_transposed_hadamard_input(
            bytes,
            scales,
            source_global_scales,
            chunk_base,
            lane,
            source_rows,
            source_cols,
            dst_row_len,
            sign_seed,
        );
        ms_eden_pack_chunk(
            input,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            &mut out_chunk_amax,
            chunk,
            dst_row_len,
            global_scale[0],
            scale_override,
            scale_seed,
        );
    }

    #[kernel]
    pub fn quartet_backward_ms_eden_global_scale_from_chunks_kernel(
        chunk_amax: &[f32],
        mut out_global_scale: DisjointSlice<f32>,
        chunk_count: u32,
    ) {
        let thread = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = thread / 32;
        let mut chunk = thread;
        let mut local_amax = 0.0;
        let stride = thread::blockDim_x();

        while chunk < chunk_count {
            local_amax = max_f32(
                local_amax,
                max4_f32(
                    chunk_amax_or_zero(chunk_amax, chunk, chunk_count),
                    chunk_amax_or_zero(chunk_amax, chunk + stride, chunk_count),
                    chunk_amax_or_zero(chunk_amax, chunk + stride * 2, chunk_count),
                    chunk_amax_or_zero(chunk_amax, chunk + stride * 3, chunk_count),
                ),
            );
            chunk += stride * 4;
        }

        let warp_amax = warp_max_f32(local_amax);
        if lane == 0 {
            unsafe {
                AMAX_REDUCE[warp_in_block as usize] = warp_amax;
            }
        }

        thread::sync_threads();

        if warp_in_block == 0 {
            let partial = if lane < AMAX_WARPS_PER_BLOCK {
                unsafe { AMAX_REDUCE[lane as usize] }
            } else {
                0.0
            };
            let amax = warp_max_f32(partial);
            if lane == 0 {
                let global_scale = if amax == 0.0 {
                    1.0
                } else {
                    amax * QUARTET_MS_EDEN_SCALE_OVERRIDE
                        / (QUARTET_MS_EDEN_FP8_MAX * QUARTET_MS_EDEN_FP4_MAX)
                };
                unsafe {
                    *out_global_scale.get_unchecked_mut(0) = global_scale;
                }
            }
        }
    }

    #[inline(always)]
    fn chunk_amax_or_zero(chunk_amax: &[f32], chunk: u32, chunk_count: u32) -> f32 {
        if chunk < chunk_count {
            chunk_amax[chunk as usize]
        } else {
            0.0
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn fp32_to_nvfp4_ms_eden_body(
        x: &[f32],
        out_fp4: &mut DisjointSlice<'_, u8>,
        out_scales: &mut DisjointSlice<'_, u8>,
        out_global_scales: &mut DisjointSlice<'_, f32>,
        out_chunk_amax: &mut DisjointSlice<'_, f32>,
        chunk: u32,
        src_row_len: u32,
        dst_row_len: u32,
        global_scale: f32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let lane = warp::lane_id();
        let chunk_base = chunk * HADAMARD_DIM;

        let input = hadamard_input(x, chunk_base, lane, src_row_len, dst_row_len, sign_seed);
        ms_eden_pack_chunk(
            input,
            out_fp4,
            out_scales,
            out_global_scales,
            out_chunk_amax,
            chunk,
            dst_row_len,
            global_scale,
            scale_override,
            scale_seed,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn fp32_transpose_to_nvfp4_ms_eden_body(
        x: &[f32],
        out_fp4: &mut DisjointSlice<'_, u8>,
        out_scales: &mut DisjointSlice<'_, u8>,
        out_global_scales: &mut DisjointSlice<'_, f32>,
        out_chunk_amax: &mut DisjointSlice<'_, f32>,
        chunk: u32,
        source_rows: u32,
        dst_row_len: u32,
        source_cols: u32,
        global_scale: f32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let lane = warp::lane_id();
        let chunk_base = chunk * HADAMARD_DIM;

        let input = transposed_hadamard_input(
            x,
            chunk_base,
            lane,
            source_rows,
            dst_row_len,
            source_cols,
            sign_seed,
        );
        ms_eden_pack_chunk(
            input,
            out_fp4,
            out_scales,
            out_global_scales,
            out_chunk_amax,
            chunk,
            dst_row_len,
            global_scale,
            scale_override,
            scale_seed,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn ms_eden_pack_chunk(
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
        let inv_scale = 1.0 / (scale * safe_global_scale);
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
    fn pack_chunk() -> u32 {
        thread::blockIdx_x() * AMAX_WARPS_PER_BLOCK + thread::threadIdx_x() / 32
    }

    #[inline(always)]
    fn nvfp4_value_at(bytes: &[u8], scales: &[u8], global_scale: &[f32], index: u32) -> f32 {
        nvfp4_value(bytes, scales, global_scale[0], index as usize)
    }

    #[inline(always)]
    fn checked_nvfp4_abs_value(
        bytes: &[u8],
        scales: &[u8],
        global_scale: &[f32],
        index: u32,
        element_count: u32,
    ) -> f32 {
        if index < element_count {
            abs_f32(nvfp4_value_at(bytes, scales, global_scale, index))
        } else {
            0.0
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn nvfp4_transposed_hadamard_input(
        bytes: &[u8],
        scales: &[u8],
        global_scale: &[f32],
        chunk_base: u32,
        lane: u32,
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        seed: u32,
    ) -> f32 {
        let row = chunk_base / dst_row_len;
        let row_base = row * dst_row_len;
        let chunk_in_row = chunk_base - row_base;
        let input_col = chunk_in_row + lane;
        let input = if input_col < source_rows {
            let source_index = input_col * source_cols + row;
            nvfp4_value_at(bytes, scales, global_scale, source_index)
        } else {
            0.0
        };
        input * random_sign(seed, input_col)
    }

    #[inline(always)]
    fn rowwise_value_at(
        bytes: &[u8],
        scales: &[u8],
        global_scales: &[f32],
        cols: u32,
        index: u32,
    ) -> f32 {
        let row = index / cols;
        let col = index - row * cols;
        nvfp4_rowwise_value(
            bytes,
            scales,
            global_scales,
            cols as usize,
            row as usize,
            col as usize,
        )
    }

    #[inline(always)]
    fn checked_rowwise_abs_value(
        bytes: &[u8],
        scales: &[u8],
        global_scales: &[f32],
        cols: u32,
        index: u32,
        element_count: u32,
    ) -> f32 {
        if index < element_count {
            abs_f32(rowwise_value_at(bytes, scales, global_scales, cols, index))
        } else {
            0.0
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn rowwise_transposed_hadamard_input(
        bytes: &[u8],
        scales: &[u8],
        global_scales: &[f32],
        chunk_base: u32,
        lane: u32,
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        seed: u32,
    ) -> f32 {
        let row = chunk_base / dst_row_len;
        let row_base = row * dst_row_len;
        let chunk_in_row = chunk_base - row_base;
        let input_col = chunk_in_row + lane;
        let input = if input_col < source_rows {
            nvfp4_rowwise_value(
                bytes,
                scales,
                global_scales,
                source_cols as usize,
                input_col as usize,
                row as usize,
            )
        } else {
            0.0
        };
        input * random_sign(seed, input_col)
    }

    #[inline(always)]
    fn hadamard_input(
        x: &[f32],
        chunk_base: u32,
        lane: u32,
        src_row_len: u32,
        dst_row_len: u32,
        seed: u32,
    ) -> f32 {
        let row = chunk_base / dst_row_len;
        let row_base = row * dst_row_len;
        let chunk_in_row = chunk_base - row_base;
        let input_col = chunk_in_row + lane;
        let input = if input_col < src_row_len {
            let index = row * src_row_len + input_col;
            x[index as usize]
        } else {
            0.0
        };
        input * random_sign(seed, input_col)
    }

    #[inline(always)]
    fn transposed_hadamard_input(
        x: &[f32],
        chunk_base: u32,
        lane: u32,
        source_rows: u32,
        dst_row_len: u32,
        source_cols: u32,
        seed: u32,
    ) -> f32 {
        let row = chunk_base / dst_row_len;
        let row_base = row * dst_row_len;
        let chunk_in_row = chunk_base - row_base;
        let input_col = chunk_in_row + lane;
        let input = if input_col < source_rows {
            let index = input_col * source_cols + row;
            x[index as usize]
        } else {
            0.0
        };
        input * random_sign(seed, input_col)
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
