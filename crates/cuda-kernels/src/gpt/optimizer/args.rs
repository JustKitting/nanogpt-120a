use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AuroraSlotDescriptor {
    pub grad: u64,
    pub momentum: u64,
    pub z_master: u64,
    pub x_master: u64,
    pub bytes: u64,
    pub scales: u64,
    pub global_scale: u64,
    pub rows: u32,
    pub cols: u32,
    pub learning_rate_multiplier: f32,
}

unsafe impl DeviceCopy for AuroraSlotDescriptor {}

pub struct AuroraMegaUpdateArgs<'a> {
    pub stream: &'a CudaStream,
    pub slots: &'a DeviceBuffer<AuroraSlotDescriptor>,
    pub oriented: &'a mut DeviceBuffer<f32>,
    pub polar_next: &'a mut DeviceBuffer<f32>,
    pub polar_x: &'a mut DeviceBuffer<f32>,
    pub polar_gram: &'a mut DeviceBuffer<f32>,
    pub polar_ax: &'a mut DeviceBuffer<f32>,
    pub polar_chunks: &'a mut DeviceBuffer<f32>,
    pub slot_count: u32,
    pub max_len: u32,
    pub max_ax_len: u32,
    pub max_dim: u32,
    pub mu: f32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub average_coefficient: f32,
    pub iterations: u32,
}

pub struct AdamWUpdateArgs<'a> {
    pub stream: &'a CudaStream,
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scale: &'a mut DeviceBuffer<f32>,
    pub z_master: &'a mut DeviceBuffer<f32>,
    pub x_master: &'a mut DeviceBuffer<f32>,
    pub grad: &'a DeviceBuffer<f32>,
    pub first_moment: &'a mut DeviceBuffer<f32>,
    pub second_moment: &'a mut DeviceBuffer<f32>,
    pub amax: &'a mut DeviceBuffer<f32>,
    pub chunk_amax: &'a mut DeviceBuffer<f32>,
    pub len: u32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub beta1_correction: f32,
    pub beta2_correction: f32,
    pub eps: f32,
    pub average_coefficient: f32,
}

pub struct ScheduleFreeMaterializeArgs<'a> {
    pub stream: &'a CudaStream,
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scale: &'a mut DeviceBuffer<f32>,
    pub z_master: &'a DeviceBuffer<f32>,
    pub x_master: &'a DeviceBuffer<f32>,
    pub amax: &'a mut DeviceBuffer<f32>,
    pub chunk_amax: &'a mut DeviceBuffer<f32>,
    pub len: u32,
    pub beta: f32,
}

pub struct EmbeddingLookupGradArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub tokens: &'a DeviceBuffer<u32>,
    pub d_embedding_residual: &'a DeviceBuffer<f32>,
    pub d_token_embedding: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub embedding_dim: u32,
}

pub struct GradientClipArgs<'a> {
    pub stream: &'a CudaStream,
    pub ptrs: &'a DeviceBuffer<u64>,
    pub lens: &'a DeviceBuffer<u32>,
    pub chunk_offsets: &'a DeviceBuffer<u32>,
    pub chunk_sums: &'a mut DeviceBuffer<f32>,
    pub scale: &'a mut DeviceBuffer<f32>,
    pub norm: &'a mut DeviceBuffer<f32>,
    pub slot_count: u32,
    pub chunk_count: u32,
    pub max_norm: f32,
}

pub struct KdaAuroraClipArgs<'a> {
    pub stream: &'a CudaStream,
    pub qkv: &'a DeviceBuffer<u16>,
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scale: &'a mut DeviceBuffer<f32>,
    pub z_master: &'a mut DeviceBuffer<f32>,
    pub x_master: &'a mut DeviceBuffer<f32>,
    pub momentum: &'a mut DeviceBuffer<f32>,
    pub scores: &'a mut DeviceBuffer<f32>,
    pub amax: &'a mut DeviceBuffer<f32>,
    pub chunk_amax: &'a mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub qkv_dim: u32,
    pub input_dim: u32,
    pub embedding_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
    pub tau: f32,
    pub silu_qk: u32,
}
