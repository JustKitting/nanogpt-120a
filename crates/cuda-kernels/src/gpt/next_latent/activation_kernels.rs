use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::float_ptx::exp_f32;

const THREADS_PER_BLOCK: u32 = 256;

#[cuda_module]
pub mod module {
    use super::*;

    #[kernel]
    pub fn nextlat_gelu_kernel(input: &[f32], mut out: DisjointSlice<f32>, len: u32) {
        let index = thread::blockIdx_x() * THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            unsafe {
                *out.get_unchecked_mut(index as usize) = gelu_f32(input[index as usize]);
            }
        }
    }

    #[kernel]
    pub fn nextlat_residual_add_kernel(
        delta: &[f32],
        residual: &[f32],
        mut out: DisjointSlice<f32>,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            unsafe {
                *out.get_unchecked_mut(index as usize) =
                    delta[index as usize] + residual[index as usize];
            }
        }
    }

    #[inline(always)]
    fn gelu_f32(x: f32) -> f32 {
        let cubic = x * x * x;
        let inner = 0.797_884_6 * (x + 0.044_715 * cubic);
        0.5 * x * (1.0 + tanh_f32(inner))
    }

    #[inline(always)]
    fn tanh_f32(x: f32) -> f32 {
        2.0 / (1.0 + exp_f32(-2.0 * x)) - 1.0
    }
}
