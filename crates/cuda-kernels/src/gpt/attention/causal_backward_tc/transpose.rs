use cuda_device::{DisjointSlice, thread};

use super::gather::TC_BACKWARD_THREADS_PER_BLOCK;

pub(super) fn transpose_body(
    src: &[f32],
    mut dst: DisjointSlice<f32>,
    batch_count: u32,
    rows: u32,
    cols: u32,
) {
    let index = thread::blockIdx_x() * TC_BACKWARD_THREADS_PER_BLOCK + thread::threadIdx_x();
    let total = batch_count * rows * cols;
    if index >= total {
        return;
    }

    let col = index % cols;
    let row = (index / cols) % rows;
    let batch = index / (rows * cols);
    let dst_index = batch * rows * cols + col * rows + row;
    unsafe {
        *dst.get_unchecked_mut(dst_index as usize) = src[index as usize];
    }
}
