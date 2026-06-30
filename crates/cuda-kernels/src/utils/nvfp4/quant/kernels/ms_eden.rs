use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::amax::{amax4_f32, max4_f32};
use crate::block_reduce::block_max_leader_f32;
use crate::float_ptx::max_f32;
use crate::nvfp4::nvfp4_rowwise_value;
use crate::quartet::quartet_backward_ms_eden_global_scale;

use super::row_amax::TENSOR_AMAX_VALUES_PER_BLOCK;

const HADAMARD_DIM: u32 = 32;
const GROUP_SIZE: u32 = 16;
const INV_SQRT_32: f32 = 0.176_776_69;
const FP4_MAX: f32 = 6.0;
const AMAX_WARPS_PER_BLOCK: u32 = crate::nvfp4_quant::config::WARPS_PER_BLOCK;

#[path = "ms_eden/body.rs"]
mod body;
#[path = "ms_eden/input.rs"]
mod input;
#[path = "ms_eden/pack.rs"]
mod pack;
#[path = "ms_eden/random.rs"]
mod random;

#[allow(static_mut_refs)]
#[cuda_module]
pub(crate) mod module {
    use super::body::*;
    use super::input::*;
    use super::pack::*;
    use super::random::random_sign;
    use super::*;

    static mut AMAX_REDUCE: SharedArray<f32, { AMAX_WARPS_PER_BLOCK as usize }> =
        SharedArray::UNINIT;

    macro_rules! guarded_pack_chunk {
        ($chunk:ident, $chunk_count:ident) => {
            let $chunk = pack_chunk();
            if $chunk >= $chunk_count {
                return;
            }
        };
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
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
        guarded_pack_chunk!(chunk, chunk_count);

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
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
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
        guarded_pack_chunk!(chunk, chunk_count);

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
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn fp32_to_nvfp4_ms_eden_device_scale_no_chunk_amax_kernel(
        x: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        chunk_count: u32,
        src_row_len: u32,
        dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        guarded_pack_chunk!(chunk, chunk_count);

        fp32_to_nvfp4_ms_eden_body_no_chunk_amax(
            x,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
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
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn fp32_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_kernel(
        x: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        src_row_len: u32,
        dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        fp32_to_nvfp4_ms_eden_body_no_chunk_amax(
            x,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            pack_chunk(),
            src_row_len,
            dst_row_len,
            global_scale[0],
            scale_override,
            sign_seed,
            scale_seed,
        );
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
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
        guarded_pack_chunk!(chunk, chunk_count);

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
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn fp32_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_kernel(
        x: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        chunk_count: u32,
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        guarded_pack_chunk!(chunk, chunk_count);

        fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax(
            x,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
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
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn fp32_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_kernel(
        x: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax(
            x,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            pack_chunk(),
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

        if let Some(block_amax) =
            unsafe { block_max_leader_f32(&mut AMAX_REDUCE, local_amax, lane, warp_in_block) }
        {
            unsafe {
                *out.get_unchecked_mut(chunk as usize) = block_amax;
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

        if let Some(block_amax) =
            unsafe { block_max_leader_f32(&mut AMAX_REDUCE, local_amax, lane, warp_in_block) }
        {
            unsafe {
                *out.get_unchecked_mut(chunk as usize) = block_amax;
            }
        }
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
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
        guarded_pack_chunk!(chunk, chunk_count);

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
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scale: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
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
        guarded_pack_chunk!(chunk, chunk_count);

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
        ms_eden_pack_chunk_no_chunk_amax(
            input,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            chunk,
            dst_row_len,
            global_scale[0],
            scale_override,
            scale_seed,
        );
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scale: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let lane = warp::lane_id();
        let chunk = pack_chunk();
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
        ms_eden_pack_chunk_no_chunk_amax(
            input,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            chunk,
            dst_row_len,
            global_scale[0],
            scale_override,
            scale_seed,
        );
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
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
        guarded_pack_chunk!(chunk, chunk_count);

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
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scales: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
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
        guarded_pack_chunk!(chunk, chunk_count);

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
        ms_eden_pack_chunk_no_chunk_amax(
            input,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            chunk,
            dst_row_len,
            global_scale[0],
            scale_override,
            scale_seed,
        );
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scales: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let lane = warp::lane_id();
        let chunk = pack_chunk();
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
        ms_eden_pack_chunk_no_chunk_amax(
            input,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            chunk,
            dst_row_len,
            global_scale[0],
            scale_override,
            scale_seed,
        );
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scales: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        source_cols: u32,
        chunks_per_row_shift: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let lane = warp::lane_id();
        let chunk = pack_chunk();
        let chunk_in_row_mask = (1u32 << chunks_per_row_shift) - 1;
        let row = chunk >> chunks_per_row_shift;
        let chunk_in_row = (chunk & chunk_in_row_mask) * HADAMARD_DIM;
        let input_col = chunk_in_row + lane;
        let input = nvfp4_rowwise_value(
            bytes,
            scales,
            source_global_scales,
            source_cols as usize,
            input_col as usize,
            row as usize,
        ) * random_sign(sign_seed, input_col);
        ms_eden_pack_chunk_no_chunk_amax_row(
            input,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            chunk,
            row,
            chunk_in_row == 0,
            global_scale[0],
            scale_override,
            scale_seed,
        );
    }

    #[kernel]
    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_source_cols_pow2_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scales: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        source_cols_shift: u32,
        chunks_per_row_shift: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let lane = warp::lane_id();
        let chunk = pack_chunk();
        let chunk_in_row_mask = (1u32 << chunks_per_row_shift) - 1;
        let row = chunk >> chunks_per_row_shift;
        let chunk_in_row = (chunk & chunk_in_row_mask) * HADAMARD_DIM;
        let input_col = chunk_in_row + lane;
        let input = nvfp4_rowwise_value_at_pow2(
            bytes,
            scales,
            source_global_scales,
            source_cols_shift,
            input_col,
            row,
        ) * random_sign(sign_seed, input_col);
        ms_eden_pack_chunk_no_chunk_amax_row(
            input,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            chunk,
            row,
            chunk_in_row == 0,
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
}
