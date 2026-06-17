use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::atomic::atomic_add_f32;

use super::EMBEDDING_GRAD_THREADS_PER_BLOCK;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn embedding_lookup_grad_add_kernel(
        tokens: &[u32],
        d_embedding_residual: &[f32],
        mut d_token_embedding: DisjointSlice<f32>,
        token_count: u32,
        embedding_dim: u32,
    ) {
        let index = thread::blockIdx_x() * EMBEDDING_GRAD_THREADS_PER_BLOCK + thread::threadIdx_x();
        let len = token_count * embedding_dim;

        if index < len {
            let row = index / embedding_dim;
            let col = index - row * embedding_dim;
            let token = tokens[row as usize];
            let dst = (token * embedding_dim + col) as usize;
            let value = d_embedding_residual[index as usize];

            unsafe {
                atomic_add_f32(d_token_embedding.as_mut_ptr().add(dst), value);
            }
        }
    }
}
