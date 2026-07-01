use cuda_core::DeviceCopy;

const THREADS_PER_BLOCK: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ApplyRopeParams {
    pub row_count: u32,
    pub seq_len: u32,
    pub batch_size: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
}

unsafe impl DeviceCopy for ApplyRopeParams {}

#[path = "rope/body.rs"]
mod body;
#[path = "rope/kernels.rs"]
pub mod kernels;
#[path = "rope/launcher.rs"]
mod launcher;
pub use launcher::ApplyRopeArgs;
