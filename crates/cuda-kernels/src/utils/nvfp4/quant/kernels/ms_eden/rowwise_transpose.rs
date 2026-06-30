use cuda_device::{DisjointSlice, cuda_module, kernel, warp};

use crate::nvfp4::nvfp4_rowwise_value;

use super::HADAMARD_DIM;
use super::input::{
    no_pad_pow2_chunk_position, nvfp4_rowwise_value_at_pow2, rowwise_transposed_hadamard_input,
};
use super::pack::{
    guarded_pack_chunk, ms_eden_pack_chunk, ms_eden_pack_chunk_no_chunk_amax,
    ms_eden_pack_chunk_no_chunk_amax_row, pack_chunk,
};
use super::random::random_sign;
use super::transpose_kernels::pack_padded_transpose_chunk;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[cuda_module]
pub(crate) mod module {
    use super::*;

    #[kernel]
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
        guarded_pack_chunk!(chunk, chunk_count);
        pack_padded_transpose_chunk!(
            chunk_amax,
            input: rowwise_transposed_hadamard_input(bytes, scales, source_global_scales),
            chunk: chunk,
            output: [
                &mut out_fp4,
                &mut out_scales,
                &mut out_global_scales,
                &mut out_chunk_amax
            ],
            dims: [source_rows, source_cols, dst_row_len],
            scale: [global_scale[0], scale_override, scale_seed],
            sign_seed: sign_seed,
        );
    }

    #[kernel]
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
        guarded_pack_chunk!(chunk, chunk_count);
        pack_padded_transpose_chunk!(
            no_chunk_amax,
            input: rowwise_transposed_hadamard_input(bytes, scales, source_global_scales),
            chunk: chunk,
            output: [&mut out_fp4, &mut out_scales, &mut out_global_scales],
            dims: [source_rows, source_cols, dst_row_len],
            scale: [global_scale[0], scale_override, scale_seed],
            sign_seed: sign_seed,
        );
    }

    #[kernel]
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
        pack_padded_transpose_chunk!(
            no_chunk_amax,
            input: rowwise_transposed_hadamard_input(bytes, scales, source_global_scales),
            chunk: pack_chunk(),
            output: [&mut out_fp4, &mut out_scales, &mut out_global_scales],
            dims: [source_rows, source_cols, dst_row_len],
            scale: [global_scale[0], scale_override, scale_seed],
            sign_seed: sign_seed,
        );
    }

    #[kernel]
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
        let (row, input_col, first_chunk_in_row) =
            no_pad_pow2_chunk_position(chunk, lane, chunks_per_row_shift);
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
            first_chunk_in_row,
            global_scale[0],
            scale_override,
            scale_seed,
        );
    }

    #[kernel]
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
        let (row, input_col, first_chunk_in_row) =
            no_pad_pow2_chunk_position(chunk, lane, chunks_per_row_shift);
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
            first_chunk_in_row,
            global_scale[0],
            scale_override,
            scale_seed,
        );
    }
}
