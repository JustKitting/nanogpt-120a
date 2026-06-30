use cuda_core::DeviceCopy;

const CROSS_ENTROPY_THREADS_PER_BLOCK: u32 = 1024;
const WARP_SIZE: u32 = 32;
const CROSS_ENTROPY_WARPS_PER_BLOCK: u32 = CROSS_ENTROPY_THREADS_PER_BLOCK / WARP_SIZE;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CrossEntropyParams {
    pub token_count: u32,
    pub vocab_size: u32,
}

unsafe impl DeviceCopy for CrossEntropyParams {}

#[path = "loss/kernels.rs"]
mod kernels;
#[path = "loss/launcher.rs"]
mod launcher;
pub use launcher::{CrossEntropyArgs, LossModule};
