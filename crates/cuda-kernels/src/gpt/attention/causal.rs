use cuda_core::DeviceCopy;

pub(crate) const CAUSAL_ATTENTION_MAX_THREADS_PER_BLOCK: u32 = 128;
pub(crate) const CAUSAL_MAX_WARPS_PER_BLOCK: u32 = CAUSAL_ATTENTION_MAX_THREADS_PER_BLOCK / 32;
const CAUSAL_CHUNK_SIZE: u32 = 64;
const CAUSAL_DECAY_SCALE: f32 = 0.01;

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

impl CausalAttentionParams {
    pub(crate) fn new(
        row_count: u32,
        seq_len: u32,
        batch_size: u32,
        embedding_dim: u32,
        qkv_dim: u32,
        head_count: u32,
        head_dim: u32,
    ) -> Self {
        Self {
            row_count,
            seq_len,
            batch_size,
            embedding_dim,
            qkv_dim,
            head_count,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            chunk_size: CAUSAL_CHUNK_SIZE,
            decay_scale: CAUSAL_DECAY_SCALE,
        }
    }
}

#[path = "causal/kernels.rs"]
pub mod kernels;
#[path = "causal/launcher.rs"]
mod launcher;
pub use launcher::CausalAttentionArgs;
