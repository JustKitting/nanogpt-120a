use cuda_device::{DisjointSlice, warp};

use super::HADAMARD_DIM;
use super::input::{
    hadamard_input, hadamard_input_no_pad, hadamard_input_no_pad_pow2, transposed_hadamard_input,
    transposed_hadamard_input_no_pad, transposed_hadamard_input_no_pad_pow2,
};
use super::pack::{
    ms_eden_pack_chunk, ms_eden_pack_chunk_no_chunk_amax, ms_eden_pack_chunk_no_chunk_amax_row,
};

macro_rules! ms_eden_padded_body {
    ($name:ident, $input_fn:ident, [$($input_arg:ident),+], $dst_row_len:ident, chunk_amax) => {
        #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
        #[inline(always)]
        pub(super) fn $name(
            x: &[f32],
            out_fp4: &mut DisjointSlice<'_, u8>,
            out_scales: &mut DisjointSlice<'_, u8>,
            out_global_scales: &mut DisjointSlice<'_, f32>,
            out_chunk_amax: &mut DisjointSlice<'_, f32>,
            chunk: u32,
            $($input_arg: u32,)+
            global_scale: f32,
            scale_override: f32,
            sign_seed: u32,
            scale_seed: u32,
        ) {
            let lane = warp::lane_id();
            let chunk_base = chunk * HADAMARD_DIM;
            let input = $input_fn(x, chunk_base, lane, $($input_arg,)+ sign_seed);

            ms_eden_pack_chunk(
                input,
                out_fp4,
                out_scales,
                out_global_scales,
                out_chunk_amax,
                chunk,
                $dst_row_len,
                global_scale,
                scale_override,
                scale_seed,
            );
        }
    };
    ($name:ident, $input_fn:ident, [$($input_arg:ident),+], $dst_row_len:ident, no_chunk_amax) => {
        #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
        #[inline(always)]
        pub(super) fn $name(
            x: &[f32],
            out_fp4: &mut DisjointSlice<'_, u8>,
            out_scales: &mut DisjointSlice<'_, u8>,
            out_global_scales: &mut DisjointSlice<'_, f32>,
            chunk: u32,
            $($input_arg: u32,)+
            global_scale: f32,
            scale_override: f32,
            sign_seed: u32,
            scale_seed: u32,
        ) {
            let lane = warp::lane_id();
            let chunk_base = chunk * HADAMARD_DIM;
            let input = $input_fn(x, chunk_base, lane, $($input_arg,)+ sign_seed);

            ms_eden_pack_chunk_no_chunk_amax(
                input,
                out_fp4,
                out_scales,
                out_global_scales,
                chunk,
                $dst_row_len,
                global_scale,
                scale_override,
                scale_seed,
            );
        }
    };
}

macro_rules! ms_eden_row_body {
    ($name:ident, $input_fn:ident, $row_len_arg:ident, $chunks_arg:ident) => {
        #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
        #[inline(always)]
        pub(super) fn $name(
            x: &[f32],
            out_fp4: &mut DisjointSlice<'_, u8>,
            out_scales: &mut DisjointSlice<'_, u8>,
            out_global_scales: &mut DisjointSlice<'_, f32>,
            chunk: u32,
            $row_len_arg: u32,
            $chunks_arg: u32,
            global_scale: f32,
            scale_override: f32,
            sign_seed: u32,
            scale_seed: u32,
        ) {
            let lane = warp::lane_id();
            let (input, row, first_chunk_in_row) =
                $input_fn(x, chunk, lane, $row_len_arg, $chunks_arg, sign_seed);

            ms_eden_pack_chunk_no_chunk_amax_row(
                input,
                out_fp4,
                out_scales,
                out_global_scales,
                chunk,
                row,
                first_chunk_in_row,
                global_scale,
                scale_override,
                scale_seed,
            );
        }
    };
}

ms_eden_padded_body!(
    fp32_to_nvfp4_ms_eden_body,
    hadamard_input,
    [src_row_len, dst_row_len],
    dst_row_len,
    chunk_amax
);
ms_eden_padded_body!(
    fp32_to_nvfp4_ms_eden_body_no_chunk_amax,
    hadamard_input,
    [src_row_len, dst_row_len],
    dst_row_len,
    no_chunk_amax
);
ms_eden_padded_body!(
    fp32_transpose_to_nvfp4_ms_eden_body,
    transposed_hadamard_input,
    [source_rows, dst_row_len, source_cols],
    dst_row_len,
    chunk_amax
);
ms_eden_padded_body!(
    fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax,
    transposed_hadamard_input,
    [source_rows, dst_row_len, source_cols],
    dst_row_len,
    no_chunk_amax
);

ms_eden_row_body!(
    fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad_pow2,
    hadamard_input_no_pad_pow2,
    src_row_len,
    chunks_per_row_shift
);
ms_eden_row_body!(
    fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad_pow2,
    transposed_hadamard_input_no_pad_pow2,
    source_cols,
    chunks_per_row_shift
);
ms_eden_row_body!(
    fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad,
    hadamard_input_no_pad,
    src_row_len,
    chunks_per_row
);
ms_eden_row_body!(
    fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad,
    transposed_hadamard_input_no_pad,
    source_cols,
    chunks_per_row
);
