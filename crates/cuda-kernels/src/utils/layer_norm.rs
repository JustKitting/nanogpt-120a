use cuda_device::DisjointSlice;

use crate::float_ptx::{abs_f32, fma_f32, max_f32};
use crate::nvfp4::nvfp4_value;

#[inline(always)]
pub fn f32_column(values: &[f32], row_base: usize, col: u32, row_len: u32) -> f32 {
    if col < row_len {
        values[row_base + col as usize]
    } else {
        0.0
    }
}

#[inline(always)]
pub fn nvfp4_column(
    bytes: &[u8],
    scales: &[u8],
    global_scale: f32,
    row_base: usize,
    col: u32,
    row_len: u32,
) -> f32 {
    if col < row_len {
        nvfp4_value(bytes, scales, global_scale, row_base + col as usize)
    } else {
        0.0
    }
}

#[inline(always)]
pub fn centered_column(col: u32, row_len: u32, value: f32, mean: f32) -> f32 {
    if col < row_len { value - mean } else { 0.0 }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn nvfp4_affine_normalized_column(
    weight_bytes: &[u8],
    weight_scales: &[u8],
    bias_bytes: &[u8],
    bias_scales: &[u8],
    col: u32,
    row_len: u32,
    centered: f32,
    inv_std: f32,
    weight_global_scale: f32,
    bias_global_scale: f32,
) -> f32 {
    if col < row_len {
        let weight = nvfp4_value(
            weight_bytes,
            weight_scales,
            weight_global_scale,
            col as usize,
        );
        let bias = nvfp4_value(bias_bytes, bias_scales, bias_global_scale, col as usize);
        fma_f32(centered * inv_std, weight, bias)
    } else {
        0.0
    }
}

#[inline(always)]
pub fn store_column(
    values: &mut DisjointSlice<'_, f32>,
    row_base: usize,
    col: u32,
    row_len: u32,
    value: f32,
) {
    if col < row_len {
        unsafe {
            *values.get_unchecked_mut(row_base + col as usize) = value;
        }
    }
}

#[inline(always)]
pub fn max_abs3(a: f32, b: f32, c: f32) -> f32 {
    max_f32(abs_f32(a), max_f32(abs_f32(b), abs_f32(c)))
}
