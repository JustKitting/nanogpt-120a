use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::float_ptx::fma_f32;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn f32_linear2_kernel(
        a: &[f32],
        b: &[f32],
        mut out: DisjointSlice<f32>,
        len: u32,
        a_scale: f32,
        b_scale: f32,
    ) {
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let stride = thread::gridDim_x() * thread::blockDim_x();
        while index < len {
            let i = index as usize;
            unsafe {
                *out.get_unchecked_mut(i) = fma_f32(a_scale, a[i], b_scale * b[i]);
            }
            index += stride;
        }
    }

    #[kernel]
    pub fn f32_add_scaled_identity_kernel(
        src: &[f32],
        mut out: DisjointSlice<f32>,
        dim: u32,
        scale: f32,
    ) {
        let len = dim * dim;
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let stride = thread::gridDim_x() * thread::blockDim_x();
        while index < len {
            let add = if index.is_multiple_of(dim + 1) { scale } else { 0.0 };
            unsafe {
                *out.get_unchecked_mut(index as usize) = src[index as usize] + add;
            }
            index += stride;
        }
    }
}
