use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use super::APPLY_THREADS_PER_BLOCK;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn fp32_weight_update_kernel(
        mut master: DisjointSlice<f32>,
        aurora_update: &[f32],
        learning_rate: f32,
        weight_decay: f32,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * APPLY_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            let i = index as usize;
            unsafe {
                let master = master.as_mut_ptr().add(i);
                let current = *master;
                let decay = 1.0 - learning_rate * weight_decay;
                let next = current * decay - learning_rate * aurora_update[i];
                *master = next;
            }
        }
    }
}
