use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use super::threads::APPLY_THREADS_PER_BLOCK;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn schedule_free_interpolate_kernel(
        z_master: &[f32],
        x_master: &[f32],
        mut out: DisjointSlice<f32>,
        beta: f32,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * APPLY_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            let i = index as usize;
            let z = z_master[i];
            let x = x_master[i];
            unsafe {
                *out.get_unchecked_mut(i) = z + beta * (x - z);
            }
        }
    }

    #[kernel]
    pub fn schedule_free_average_kernel(
        mut x_master: DisjointSlice<f32>,
        z_master: &[f32],
        coefficient: f32,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * APPLY_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            let i = index as usize;
            unsafe {
                let x_master = x_master.as_mut_ptr().add(i);
                let x = *x_master;
                let z = z_master[i];
                *x_master = x + coefficient * (z - x);
            }
        }
    }
}
