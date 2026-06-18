use cuda_device::{DisjointSlice, thread};

use super::super::super::threads::MATRIX_THREADS_PER_BLOCK;

#[inline(always)]
pub(super) fn elementwise_linear_combination(
    a: &[f32],
    b: &[f32],
    out: &mut DisjointSlice<f32>,
    a_scale: f32,
    b_scale: f32,
    len: u32,
) {
    let index = thread::blockIdx_x() * MATRIX_THREADS_PER_BLOCK + thread::threadIdx_x();
    if index < len {
        unsafe {
            *out.get_unchecked_mut(index as usize) =
                a_scale * a[index as usize] + b_scale * b[index as usize];
        }
    }
}
