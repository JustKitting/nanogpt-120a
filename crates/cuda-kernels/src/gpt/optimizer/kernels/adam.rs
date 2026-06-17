use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::float_ptx::sqrt_f32;
use crate::nvfp4::nvfp4_value;

use super::APPLY_THREADS_PER_BLOCK;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn nvfp4_adamw_update_to_f32_kernel(
        bytes: &[u8],
        scales: &[u8],
        grad: &[f32],
        mut first_moment: DisjointSlice<f32>,
        mut second_moment: DisjointSlice<f32>,
        mut fp32_workspace: DisjointSlice<f32>,
        global_scale: f32,
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
                let first = first_moment.as_mut_ptr().add(i);
                let second = second_moment.as_mut_ptr().add(i);
                let m = beta1 * *first + (1.0 - beta1) * g;
                let v = beta2 * *second + (1.0 - beta2) * g * g;
                let update = (m / beta1_correction) / (sqrt_f32(v / beta2_correction) + eps);
                let current = nvfp4_value(bytes, scales, global_scale, i);
                let decay = 1.0 - learning_rate * weight_decay;
                let next = current * decay - learning_rate * update;

                *first = m;
                *second = v;
                *fp32_workspace.get_unchecked_mut(i) = next;
            }
        }
    }
}
