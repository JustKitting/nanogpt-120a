#[path = "layer_norm/columns.rs"]
mod columns;

pub use columns::{
    centered_column, f16_column, f32_column, max_abs3, nvfp4_affine_normalized_column,
    nvfp4_column, store_column, store_f16_column,
};

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
