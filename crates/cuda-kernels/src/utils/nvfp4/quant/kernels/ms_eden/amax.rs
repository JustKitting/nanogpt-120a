use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};

use crate::amax::{amax4_f32, max4_f32};
use crate::block_reduce::{block_max_leader_f32, block_max_store_f32};
use crate::float_ptx::max_f32;
use crate::quartet::quartet_backward_ms_eden_global_scale;
use crate::warp_reduce::thread_lane_warp;

use super::input::{
    checked_nvfp4_abs_value, checked_rowwise_abs_value, nvfp4_value_at, rowwise_value_at,
};
use super::AMAX_WARPS_PER_BLOCK;
use crate::nvfp4_quant::kernels::row_amax::TENSOR_AMAX_VALUES_PER_BLOCK;

#[cuda_module]
pub(crate) mod module {
    use super::*;

    static mut AMAX_REDUCE: SharedArray<f32, { AMAX_WARPS_PER_BLOCK as usize }> =
        SharedArray::UNINIT;

    #[kernel]
    pub fn rowwise_nvfp4_chunk_amax_kernel(
        bytes: &[u8],
        scales: &[u8],
        global_scales: &[f32],
        mut out: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
    ) {
        let element_count = rows * cols;
        let (chunk, lane, warp_in_block, base, i0, i1, i2, i3) = chunk_amax_indices();

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

        block_max_store_f32!(AMAX_REDUCE, out[chunk], local_amax, lane, warp_in_block);
    }

    #[kernel]
    pub fn nvfp4_chunk_amax_kernel(
        bytes: &[u8],
        scales: &[u8],
        global_scale: &[f32],
        mut out: DisjointSlice<f32>,
        element_count: u32,
    ) {
        let (chunk, lane, warp_in_block, base, i0, i1, i2, i3) = chunk_amax_indices();

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

        block_max_store_f32!(AMAX_REDUCE, out[chunk], local_amax, lane, warp_in_block);
    }

    #[kernel]
    pub fn quartet_backward_ms_eden_global_scale_from_chunks_kernel(
        chunk_amax: &[f32],
        mut out_global_scale: DisjointSlice<f32>,
        chunk_count: u32,
    ) {
        let (thread, lane, warp_in_block) = thread_lane_warp();
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

        if let Some(amax) =
            unsafe { block_max_leader_f32(&mut AMAX_REDUCE, local_amax, lane, warp_in_block) }
        {
            unsafe {
                *out_global_scale.get_unchecked_mut(0) =
                    quartet_backward_ms_eden_global_scale(amax);
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

    #[inline(always)]
    fn chunk_amax_indices() -> (u32, u32, u32, u32, u32, u32, u32, u32) {
        let chunk = thread::blockIdx_x();
        let (thread, lane, warp_in_block) = thread_lane_warp();
        let base = chunk * TENSOR_AMAX_VALUES_PER_BLOCK;
        let stride = thread::blockDim_x();
        let i0 = base + thread;
        let i1 = i0 + stride;
        let i2 = i1 + stride;
        let i3 = i2 + stride;
        (chunk, lane, warp_in_block, base, i0, i1, i2, i3)
    }
}
