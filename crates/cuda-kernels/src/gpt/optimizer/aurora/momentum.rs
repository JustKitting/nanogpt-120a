use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use super::super::threads::MATRIX_THREADS_PER_BLOCK;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn aurora_momentum_kernel(
        grad: &[f32],
        mut momentum: DisjointSlice<f32>,
        mut update: DisjointSlice<f32>,
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
                *update.get_unchecked_mut(i) = nesterov;
            }
        }
    }
}
