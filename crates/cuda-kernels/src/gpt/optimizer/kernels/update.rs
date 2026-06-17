use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::nvfp4::nvfp4_value;

use super::APPLY_THREADS_PER_BLOCK;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn nvfp4_weight_update_to_f32_kernel(
        bytes: &[u8],
        scales: &[u8],
        aurora_update: &[f32],
        mut fp32_workspace: DisjointSlice<f32>,
        global_scale: f32,
        learning_rate: f32,
        weight_decay: f32,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * APPLY_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            let current = nvfp4_value(bytes, scales, global_scale, index as usize);
            let decay = 1.0 - learning_rate * weight_decay;
            let next = current * decay - learning_rate * aurora_update[index as usize];

            unsafe {
                *fp32_workspace.get_unchecked_mut(index as usize) = next;
            }
        }
    }
}
