use rust_kernels_cuda::nvfp4_tc_matmul::nvfp4_tc_matmul_padded_k;

pub use crate::common::nvfp4_tc::Nvfp4TcScratchBuffers as ScratchBuffers;

pub const M: usize = 1;
pub const N: usize = 1;
pub const K: usize = 65;

pub fn padded_k() -> usize {
    nvfp4_tc_matmul_padded_k(K as u32) as usize
}
