use cuda_device::{DisjointSlice, thread};

use super::kernels::F16_THREADS_PER_BLOCK;

pub(super) fn pad_rows_body(
    src: &[f32],
    mut dst: DisjointSlice<f32>,
    rows: u32,
    src_cols: u32,
    dst_cols: u32,
) {
    let index = thread::blockIdx_x() * F16_THREADS_PER_BLOCK + thread::threadIdx_x();
    if index < rows * dst_cols {
        let col = index % dst_cols;
        let row = index / dst_cols;
        let value = if col < src_cols {
            src[(row * src_cols + col) as usize]
        } else {
            0.0
        };
        unsafe {
            *dst.get_unchecked_mut(index as usize) = value;
        }
    }
}
