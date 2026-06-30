use cuda_device::{DisjointSlice, warp};

use super::HADAMARD_DIM;
use super::input::{
    hadamard_input, hadamard_input_no_pad, hadamard_input_no_pad_pow2, transposed_hadamard_input,
    transposed_hadamard_input_no_pad, transposed_hadamard_input_no_pad_pow2,
};
use super::pack::{
    ms_eden_pack_chunk, ms_eden_pack_chunk_no_chunk_amax, ms_eden_pack_chunk_no_chunk_amax_row,
};

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub(super) fn fp32_to_nvfp4_ms_eden_body(
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

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub(super) fn fp32_to_nvfp4_ms_eden_body_no_chunk_amax(
    x: &[f32],
    out_fp4: &mut DisjointSlice<'_, u8>,
    out_scales: &mut DisjointSlice<'_, u8>,
    out_global_scales: &mut DisjointSlice<'_, f32>,
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
    ms_eden_pack_chunk_no_chunk_amax(
        input,
        out_fp4,
        out_scales,
        out_global_scales,
        chunk,
        dst_row_len,
        global_scale,
        scale_override,
        scale_seed,
    );
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub(super) fn fp32_transpose_to_nvfp4_ms_eden_body(
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

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub(super) fn fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax(
    x: &[f32],
    out_fp4: &mut DisjointSlice<'_, u8>,
    out_scales: &mut DisjointSlice<'_, u8>,
    out_global_scales: &mut DisjointSlice<'_, f32>,
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
    ms_eden_pack_chunk_no_chunk_amax(
        input,
        out_fp4,
        out_scales,
        out_global_scales,
        chunk,
        dst_row_len,
        global_scale,
        scale_override,
        scale_seed,
    );
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub(super) fn fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad_pow2(
    x: &[f32],
    out_fp4: &mut DisjointSlice<'_, u8>,
    out_scales: &mut DisjointSlice<'_, u8>,
    out_global_scales: &mut DisjointSlice<'_, f32>,
    chunk: u32,
    src_row_len: u32,
    chunks_per_row_shift: u32,
    global_scale: f32,
    scale_override: f32,
    sign_seed: u32,
    scale_seed: u32,
) {
    let lane = warp::lane_id();
    let (input, row, first_chunk_in_row) =
        hadamard_input_no_pad_pow2(x, chunk, lane, src_row_len, chunks_per_row_shift, sign_seed);
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

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub(super) fn fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad_pow2(
    x: &[f32],
    out_fp4: &mut DisjointSlice<'_, u8>,
    out_scales: &mut DisjointSlice<'_, u8>,
    out_global_scales: &mut DisjointSlice<'_, f32>,
    chunk: u32,
    source_cols: u32,
    chunks_per_row_shift: u32,
    global_scale: f32,
    scale_override: f32,
    sign_seed: u32,
    scale_seed: u32,
) {
    let lane = warp::lane_id();
    let (input, row, first_chunk_in_row) = transposed_hadamard_input_no_pad_pow2(
        x,
        chunk,
        lane,
        source_cols,
        chunks_per_row_shift,
        sign_seed,
    );
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

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub(super) fn fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad(
    x: &[f32],
    out_fp4: &mut DisjointSlice<'_, u8>,
    out_scales: &mut DisjointSlice<'_, u8>,
    out_global_scales: &mut DisjointSlice<'_, f32>,
    chunk: u32,
    src_row_len: u32,
    chunks_per_row: u32,
    global_scale: f32,
    scale_override: f32,
    sign_seed: u32,
    scale_seed: u32,
) {
    let lane = warp::lane_id();
    let (input, row, first_chunk_in_row) =
        hadamard_input_no_pad(x, chunk, lane, src_row_len, chunks_per_row, sign_seed);
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

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub(super) fn fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad(
    x: &[f32],
    out_fp4: &mut DisjointSlice<'_, u8>,
    out_scales: &mut DisjointSlice<'_, u8>,
    out_global_scales: &mut DisjointSlice<'_, f32>,
    chunk: u32,
    source_cols: u32,
    chunks_per_row: u32,
    global_scale: f32,
    scale_override: f32,
    sign_seed: u32,
    scale_seed: u32,
) {
    let lane = warp::lane_id();
    let (input, row, first_chunk_in_row) =
        transposed_hadamard_input_no_pad(x, chunk, lane, source_cols, chunks_per_row, sign_seed);
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
