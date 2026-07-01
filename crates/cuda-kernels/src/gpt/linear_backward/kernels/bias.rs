use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel};

use super::super::{LINEAR_BIAS_THREADS_PER_BLOCK, bias};

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn linear_bias_grad_kernel(
        e: &[f32],
        mut dbias: DisjointSlice<f32>,
        token_count: u32,
        output_dim: u32,
    ) {
        static mut LOCAL_SUMS: SharedArray<f32, { LINEAR_BIAS_THREADS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;
        bias::linear_bias_grad_body(e, &mut dbias, token_count, output_dim, unsafe {
            &mut LOCAL_SUMS
        });
    }
}

pub(super) use module::{LoadedModule, from_module};
