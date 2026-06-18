use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use super::super::threads::MATRIX_THREADS_PER_BLOCK;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn aurora_momentum_orient_kernel(
        grad: &[f32],
        mut momentum: DisjointSlice<f32>,
        mut oriented: DisjointSlice<f32>,
        mu: f32,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * MATRIX_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            let i = index as usize;
            let g = grad[i];
            unsafe {
                let momentum_ptr = momentum.as_mut_ptr().add(i);
                let next_momentum = mu * *momentum_ptr + (1.0 - mu) * g;
                let nesterov = mu * next_momentum + (1.0 - mu) * g;
                *momentum_ptr = next_momentum;
                *oriented.get_unchecked_mut(i) = nesterov;
            }
        }
    }

    #[kernel]
    pub fn aurora_momentum_orient_transpose_kernel(
        grad: &[f32],
        mut momentum: DisjointSlice<f32>,
        mut oriented: DisjointSlice<f32>,
        mu: f32,
        rows: u32,
        cols: u32,
    ) {
        let index = thread::blockIdx_x() * MATRIX_THREADS_PER_BLOCK + thread::threadIdx_x();
        let len = rows * cols;
        if index < len {
            let row = index / cols;
            let col = index - row * cols;
            let src = index as usize;
            let dst = (col * rows + row) as usize;
            let g = grad[src];
            unsafe {
                let momentum_ptr = momentum.as_mut_ptr().add(src);
                let next_momentum = mu * *momentum_ptr + (1.0 - mu) * g;
                let nesterov = mu * next_momentum + (1.0 - mu) * g;
                *momentum_ptr = next_momentum;
                *oriented.get_unchecked_mut(dst) = nesterov;
            }
        }
    }
}
