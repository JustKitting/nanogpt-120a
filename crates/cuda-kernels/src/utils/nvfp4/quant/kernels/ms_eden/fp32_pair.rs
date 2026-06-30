use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use super::AMAX_WARPS_PER_BLOCK;
use super::body::{
    fp32_to_nvfp4_ms_eden_body_no_chunk_amax, fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad,
    fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad_pow2,
    fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax,
    fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad,
    fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad_pow2,
};

#[cuda_module]
pub(crate) mod module {
    use super::*;

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn fp32_pair_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_kernel(
        x: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        mut transpose_out_fp4: DisjointSlice<u8>,
        mut transpose_out_scales: DisjointSlice<u8>,
        mut transpose_out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        row_grid_dim: u32,
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        transpose_dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
        transpose_scale_seed: u32,
    ) {
        let block = thread::blockIdx_x();
        let warp_in_block = thread::threadIdx_x() / 32;
        if block < row_grid_dim {
            let chunk = block * AMAX_WARPS_PER_BLOCK + warp_in_block;
            fp32_to_nvfp4_ms_eden_body_no_chunk_amax(
                x,
                &mut out_fp4,
                &mut out_scales,
                &mut out_global_scales,
                chunk,
                source_cols,
                dst_row_len,
                global_scale[0],
                scale_override,
                sign_seed,
                scale_seed,
            );
        } else {
            let chunk = (block - row_grid_dim) * AMAX_WARPS_PER_BLOCK + warp_in_block;
            fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax(
                x,
                &mut transpose_out_fp4,
                &mut transpose_out_scales,
                &mut transpose_out_global_scales,
                chunk,
                source_rows,
                transpose_dst_row_len,
                source_cols,
                global_scale[0],
                scale_override,
                sign_seed,
                transpose_scale_seed,
            );
        }
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn fp32_pair_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_pow2_kernel(
        x: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        mut transpose_out_fp4: DisjointSlice<u8>,
        mut transpose_out_scales: DisjointSlice<u8>,
        mut transpose_out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        row_grid_dim: u32,
        source_cols: u32,
        row_chunks_per_row_shift: u32,
        transpose_chunks_per_row_shift: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
        transpose_scale_seed: u32,
    ) {
        let block = thread::blockIdx_x();
        let warp_in_block = thread::threadIdx_x() / 32;
        if block < row_grid_dim {
            let chunk = block * AMAX_WARPS_PER_BLOCK + warp_in_block;
            fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad_pow2(
                x,
                &mut out_fp4,
                &mut out_scales,
                &mut out_global_scales,
                chunk,
                source_cols,
                row_chunks_per_row_shift,
                global_scale[0],
                scale_override,
                sign_seed,
                scale_seed,
            );
        } else {
            let chunk = (block - row_grid_dim) * AMAX_WARPS_PER_BLOCK + warp_in_block;
            fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad_pow2(
                x,
                &mut transpose_out_fp4,
                &mut transpose_out_scales,
                &mut transpose_out_global_scales,
                chunk,
                source_cols,
                transpose_chunks_per_row_shift,
                global_scale[0],
                scale_override,
                sign_seed,
                transpose_scale_seed,
            );
        }
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn fp32_pair_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_kernel(
        x: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        mut transpose_out_fp4: DisjointSlice<u8>,
        mut transpose_out_scales: DisjointSlice<u8>,
        mut transpose_out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        row_grid_dim: u32,
        source_cols: u32,
        row_chunks_per_row: u32,
        transpose_chunks_per_row: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
        transpose_scale_seed: u32,
    ) {
        let block = thread::blockIdx_x();
        let warp_in_block = thread::threadIdx_x() / 32;
        if block < row_grid_dim {
            let chunk = block * AMAX_WARPS_PER_BLOCK + warp_in_block;
            fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad(
                x,
                &mut out_fp4,
                &mut out_scales,
                &mut out_global_scales,
                chunk,
                source_cols,
                row_chunks_per_row,
                global_scale[0],
                scale_override,
                sign_seed,
                scale_seed,
            );
        } else {
            let chunk = (block - row_grid_dim) * AMAX_WARPS_PER_BLOCK + warp_in_block;
            fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad(
                x,
                &mut transpose_out_fp4,
                &mut transpose_out_scales,
                &mut transpose_out_global_scales,
                chunk,
                source_cols,
                transpose_chunks_per_row,
                global_scale[0],
                scale_override,
                sign_seed,
                transpose_scale_seed,
            );
        }
    }
}
