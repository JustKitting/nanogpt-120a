use rust_kernels_cuda::nvfp4_tc_matmul::nvfp4_tc_matmul_padded_k;

use super::{COLS, ROWS};

pub(in super::super) fn padded_rows() -> usize {
    nvfp4_tc_matmul_padded_k(ROWS as u32) as usize
}

pub(in super::super) fn input_matrix() -> Vec<f32> {
    (0..ROWS * COLS)
        .map(|index| {
            let row = index / COLS;
            let col = index % COLS;
            (row as f32 - 9.0) * 0.03125 + (col as f32 - 4.0) * 0.0078125
        })
        .collect()
}

pub(in super::super) fn cpu_amax(x: &[f32]) -> f32 {
    x.iter().fold(0.0_f32, |max, value| max.max(value.abs()))
}
