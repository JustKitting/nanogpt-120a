use crate::float_ptx::abs_f32;
use crate::nvfp4::{nvfp4_rowwise_value, nvfp4_value};
use crate::nvfp4_cast::{e2m1_value, e4m3_value};

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
