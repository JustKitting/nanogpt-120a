use cuda_device::{DisjointSlice, thread};

#[inline(always)]
pub(super) fn store_scaled(x: &[f32], out: &mut DisjointSlice<f32>, inv_norm: f32, len: u32) {
    let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();

    if index < len {
        unsafe {
            *out.get_unchecked_mut(index as usize) = x[index as usize] * inv_norm;
        }
    }
}

#[inline(always)]
pub(super) fn scale_in_place(x: &mut DisjointSlice<f32>, inv_norm: f32, len: u32) {
    let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();

    if index < len {
        unsafe {
            *x.get_unchecked_mut(index as usize) *= inv_norm;
        }
    }
}
