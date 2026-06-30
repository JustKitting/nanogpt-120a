use crate::float_ptx::abs_f32;
use crate::nvfp4::{nvfp4_rowwise_value, nvfp4_value};
use crate::nvfp4_cast::{e2m1_value, e4m3_value};

use super::HADAMARD_DIM;
use super::random::random_sign;

#[inline(always)]
pub(super) fn nvfp4_value_at(bytes: &[u8], scales: &[u8], global_scale: &[f32], index: u32) -> f32 {
    nvfp4_value(bytes, scales, global_scale[0], index as usize)
}

#[inline(always)]
pub(super) fn checked_nvfp4_abs_value(
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

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub(super) fn nvfp4_transposed_hadamard_input(
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
    let (row, input_col) = padded_chunk_position(chunk_base, lane, dst_row_len);
    let input = if input_col < source_rows {
        let source_index = input_col * source_cols + row;
        nvfp4_value_at(bytes, scales, global_scale, source_index)
    } else {
        0.0
    };
    input * random_sign(seed, input_col)
}

#[inline(always)]
pub(super) fn rowwise_value_at(
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
pub(super) fn nvfp4_rowwise_value_at_pow2(
    bytes: &[u8],
    scales: &[u8],
    global_scales: &[f32],
    row_len_shift: u32,
    row: u32,
    col: u32,
) -> f32 {
    let index = (row << row_len_shift) + col;
    let byte = bytes[(index >> 1) as usize];
    let payload = if index & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    };

    e2m1_value(payload)
        * e4m3_value(scales[(index >> 4) as usize] as u16)
        * global_scales[row as usize]
}

#[inline(always)]
pub(super) fn checked_rowwise_abs_value(
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

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub(super) fn rowwise_transposed_hadamard_input(
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
    let (row, input_col) = padded_chunk_position(chunk_base, lane, dst_row_len);
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
pub(super) fn hadamard_input(
    x: &[f32],
    chunk_base: u32,
    lane: u32,
    src_row_len: u32,
    dst_row_len: u32,
    seed: u32,
) -> f32 {
    let (row, input_col) = padded_chunk_position(chunk_base, lane, dst_row_len);
    let input = if input_col < src_row_len {
        let index = row * src_row_len + input_col;
        x[index as usize]
    } else {
        0.0
    };
    input * random_sign(seed, input_col)
}

#[inline(always)]
pub(super) fn transposed_hadamard_input(
    x: &[f32],
    chunk_base: u32,
    lane: u32,
    source_rows: u32,
    dst_row_len: u32,
    source_cols: u32,
    seed: u32,
) -> f32 {
    let (row, input_col) = padded_chunk_position(chunk_base, lane, dst_row_len);
    let input = if input_col < source_rows {
        let index = input_col * source_cols + row;
        x[index as usize]
    } else {
        0.0
    };
    input * random_sign(seed, input_col)
}

#[inline(always)]
pub(super) fn hadamard_input_no_pad_pow2(
    x: &[f32],
    chunk: u32,
    lane: u32,
    src_row_len: u32,
    chunks_per_row_shift: u32,
    seed: u32,
) -> (f32, u32, bool) {
    let pos = no_pad_pow2_chunk_position(chunk, lane, chunks_per_row_shift);
    no_pad_result(x, pos.0 * src_row_len + pos.1, seed, pos)
}

#[inline(always)]
pub(super) fn transposed_hadamard_input_no_pad_pow2(
    x: &[f32],
    chunk: u32,
    lane: u32,
    source_cols: u32,
    chunks_per_row_shift: u32,
    seed: u32,
) -> (f32, u32, bool) {
    let pos = no_pad_pow2_chunk_position(chunk, lane, chunks_per_row_shift);
    no_pad_result(x, pos.1 * source_cols + pos.0, seed, pos)
}

#[inline(always)]
pub(super) fn hadamard_input_no_pad(
    x: &[f32],
    chunk: u32,
    lane: u32,
    src_row_len: u32,
    chunks_per_row: u32,
    seed: u32,
) -> (f32, u32, bool) {
    let pos = no_pad_chunk_position(chunk, lane, chunks_per_row);
    no_pad_result(x, pos.0 * src_row_len + pos.1, seed, pos)
}

#[inline(always)]
pub(super) fn transposed_hadamard_input_no_pad(
    x: &[f32],
    chunk: u32,
    lane: u32,
    source_cols: u32,
    chunks_per_row: u32,
    seed: u32,
) -> (f32, u32, bool) {
    let pos = no_pad_chunk_position(chunk, lane, chunks_per_row);
    no_pad_result(x, pos.1 * source_cols + pos.0, seed, pos)
}

#[inline(always)]
fn padded_chunk_position(chunk_base: u32, lane: u32, dst_row_len: u32) -> (u32, u32) {
    let row = chunk_base / dst_row_len;
    let row_base = row * dst_row_len;
    let chunk_in_row = chunk_base - row_base;
    (row, chunk_in_row + lane)
}

#[inline(always)]
fn no_pad_pow2_chunk_position(
    chunk: u32,
    lane: u32,
    chunks_per_row_shift: u32,
) -> (u32, u32, bool) {
    let chunk_in_row_mask = (1u32 << chunks_per_row_shift) - 1;
    let row = chunk >> chunks_per_row_shift;
    let chunk_in_row = (chunk & chunk_in_row_mask) * HADAMARD_DIM;
    (row, chunk_in_row + lane, chunk_in_row == 0)
}

#[inline(always)]
fn no_pad_chunk_position(chunk: u32, lane: u32, chunks_per_row: u32) -> (u32, u32, bool) {
    let row = chunk / chunks_per_row;
    let chunk_in_row = (chunk - row * chunks_per_row) * HADAMARD_DIM;
    (row, chunk_in_row + lane, chunk_in_row == 0)
}

#[inline(always)]
fn no_pad_result(x: &[f32], index: u32, seed: u32, position: (u32, u32, bool)) -> (f32, u32, bool) {
    let (row, input_col, first_chunk_in_row) = position;
    (
        x[index as usize] * random_sign(seed, input_col),
        row,
        first_chunk_in_row,
    )
}
