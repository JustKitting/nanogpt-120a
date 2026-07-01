use cuda_device::{DisjointSlice, warp};

use super::super::input::{
    hadamard_input_no_pad, hadamard_input_no_pad_pow2, transposed_hadamard_input_no_pad,
    transposed_hadamard_input_no_pad_pow2,
};
use super::super::pack::ms_eden_pack_chunk_no_chunk_amax_row;

macro_rules! ms_eden_row_body {
    ($name:ident, $input_fn:ident, $row_len_arg:ident, $chunks_arg:ident) => {
        #[inline(always)]
        pub(in super::super) fn $name(
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

ms_eden_row_body!(fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad_pow2, hadamard_input_no_pad_pow2, src_row_len, chunks_per_row_shift);
ms_eden_row_body!(fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad_pow2, transposed_hadamard_input_no_pad_pow2, source_cols, chunks_per_row_shift);
ms_eden_row_body!(fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad, hadamard_input_no_pad, src_row_len, chunks_per_row);
ms_eden_row_body!(fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad, transposed_hadamard_input_no_pad, source_cols, chunks_per_row);
