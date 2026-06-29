use cuda_device::DisjointSlice;

use crate::f16_tc_matmul::convert::{cvt_f32_f16, cvt_rn_f16_f32};
use crate::float_ptx::{abs_f32, fma_f32, max_f32};
use crate::nvfp4::nvfp4_value;

macro_rules! layer_norm_columns3 {
    ($base:expr, $stride:expr) => {
        [$base, $base + $stride, $base + $stride * 2]
    };
}

macro_rules! layer_norm_map3 {
    ($cols:expr, |$col:ident| $body:expr) => {{
        layer_norm_map3!(@items $cols, |$col| $body, 0, 1, 2)
    }};
    (@items $cols:expr, |$col:ident| $body:expr, $($index:expr),+ $(,)?) => {{
        [
            $(
                {
                    let $col = $cols[$index];
                    $body
                },
            )+
        ]
    }};
}

macro_rules! layer_norm_map3_indexed {
    ($cols:expr, |$index:ident, $col:ident| $body:expr) => {{
        layer_norm_map3_indexed!(@items $cols, |$index, $col| $body, 0usize, 1usize, 2usize)
    }};
    (@items $cols:expr, |$index:ident, $col:ident| $body:expr, $($offset:expr),+ $(,)?) => {{
        [
            $(
                {
                    let $index = $offset;
                    let $col = $cols[$index];
                    $body
                },
            )+
        ]
    }};
}

macro_rules! layer_norm_sum3 {
    ($values:expr) => {
        $values[0] + $values[1] + $values[2]
    };
}

macro_rules! layer_norm_square_sum3 {
    ($values:expr) => {
        $values[0] * $values[0] + $values[1] * $values[1] + $values[2] * $values[2]
    };
}

macro_rules! layer_norm_store3 {
    ($values:expr, $row_base:expr, $cols:expr, $row_len:expr, $source:expr) => {{
        $crate::layer_norm_utils::store_column($values, $row_base, $cols[0], $row_len, $source[0]);
        $crate::layer_norm_utils::store_column($values, $row_base, $cols[1], $row_len, $source[1]);
        $crate::layer_norm_utils::store_column($values, $row_base, $cols[2], $row_len, $source[2]);
    }};
}

macro_rules! layer_norm_store_f16_3 {
    ($values:expr, $row_base:expr, $cols:expr, $row_len:expr, $source:expr) => {{
        $crate::layer_norm_utils::store_f16_column(
            $values, $row_base, $cols[0], $row_len, $source[0],
        );
        $crate::layer_norm_utils::store_f16_column(
            $values, $row_base, $cols[1], $row_len, $source[1],
        );
        $crate::layer_norm_utils::store_f16_column(
            $values, $row_base, $cols[2], $row_len, $source[2],
        );
    }};
}

pub(crate) use {
    layer_norm_columns3, layer_norm_map3, layer_norm_map3_indexed, layer_norm_square_sum3,
    layer_norm_store_f16_3, layer_norm_store3, layer_norm_sum3,
};

#[inline(always)]
pub fn f32_column(values: &[f32], row_base: usize, col: u32, row_len: u32) -> f32 {
    if col < row_len {
        values[row_base + col as usize]
    } else {
        0.0
    }
}

#[inline(always)]
pub fn f16_column(values: &[u16], row_base: usize, col: u32, row_len: u32) -> f32 {
    if col < row_len {
        cvt_f32_f16(values[row_base + col as usize])
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

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
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
pub fn store_f16_column(
    values: &mut DisjointSlice<'_, u16>,
    row_base: usize,
    col: u32,
    row_len: u32,
    value: f32,
) {
    if col < row_len {
        unsafe {
            *values.get_unchecked_mut(row_base + col as usize) = cvt_rn_f16_f32(value);
        }
    }
}

#[inline(always)]
pub fn max_abs3(a: f32, b: f32, c: f32) -> f32 {
    max_f32(abs_f32(a), max_f32(abs_f32(b), abs_f32(c)))
}
