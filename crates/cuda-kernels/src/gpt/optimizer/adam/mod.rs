use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::float_ptx::sqrt_f32;

use super::threads::APPLY_THREADS_PER_BLOCK;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn fp32_adamw_update_kernel(
        mut master: DisjointSlice<f32>,
        grad: &[f32],
        mut first_moment: DisjointSlice<f32>,
        mut second_moment: DisjointSlice<f32>,
        learning_rate: f32,
        weight_decay: f32,
        beta1: f32,
        beta2: f32,
        beta1_correction: f32,
        beta2_correction: f32,
        eps: f32,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * APPLY_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            let i = index as usize;
            let g = grad[i];

            unsafe {
                let master = master.as_mut_ptr().add(i);
                let first = first_moment.as_mut_ptr().add(i);
                let second = second_moment.as_mut_ptr().add(i);
                let m = beta1 * *first + (1.0 - beta1) * g;
                let v = beta2 * *second + (1.0 - beta2) * g * g;
                let update = (m / beta1_correction) / (sqrt_f32(v / beta2_correction) + eps);
                let current = *master;
                let decay = 1.0 - learning_rate * weight_decay;
                let next = current * decay - learning_rate * update;

                *first = m;
                *second = v;
                *master = next;
            }
        }
    }
}
