use cuda_core::DeviceCopy;

pub(crate) const CAUSAL_ATTENTION_MAX_THREADS_PER_BLOCK: u32 = 128;
pub(crate) const CAUSAL_MAX_WARPS_PER_BLOCK: u32 = CAUSAL_ATTENTION_MAX_THREADS_PER_BLOCK / 32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CausalAttentionParams {
    pub row_count: u32,
    pub seq_len: u32,
    pub batch_size: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub chunk_size: u32,
    pub decay_scale: f32,
}

unsafe impl DeviceCopy for CausalAttentionParams {}

#[path = "causal/kernels.rs"]
pub mod kernels;
#[path = "causal/launcher.rs"]
mod launcher;
pub use launcher::CausalAttentionArgs;
