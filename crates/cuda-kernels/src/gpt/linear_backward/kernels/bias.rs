use cuda_device::{cuda_module, kernel, DisjointSlice, SharedArray};

use super::super::{bias, LINEAR_BIAS_THREADS_PER_BLOCK};

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn linear_bias_grad_kernel(
        e: &[f32], mut dbias: DisjointSlice<f32>, token_count: u32, output_dim: u32,
    ) {
        static mut LOCAL_SUMS: SharedArray<f32, { LINEAR_BIAS_THREADS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;
        bias::linear_bias_grad_body(e, &mut dbias, token_count, output_dim, unsafe {
            &mut LOCAL_SUMS
        });
    }
}

pub(super) use module::{from_module, LoadedModule};
