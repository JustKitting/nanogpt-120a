use cuda_core::{CudaStream, DeviceBuffer};

pub struct Nvfp4WeightUpdateArgs<'a> {
    pub stream: &'a CudaStream,
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scale: f32,
    pub requantize_global_scale: f32,
    pub aurora_update: &'a DeviceBuffer<f32>,
    pub fp32_workspace: &'a mut DeviceBuffer<f32>,
    pub amax: &'a mut DeviceBuffer<f32>,
    pub next_global_scale: &'a mut DeviceBuffer<f32>,
    pub len: u32,
    pub learning_rate: f32,
    pub weight_decay: f32,
}

pub struct AdamWUpdateArgs<'a> {
    pub stream: &'a CudaStream,
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scale: f32,
    pub requantize_global_scale: f32,
    pub grad: &'a DeviceBuffer<f32>,
    pub first_moment: &'a mut DeviceBuffer<f32>,
    pub second_moment: &'a mut DeviceBuffer<f32>,
    pub residual: &'a mut DeviceBuffer<f32>,
    pub fp32_workspace: &'a mut DeviceBuffer<f32>,
    pub amax: &'a mut DeviceBuffer<f32>,
    pub next_global_scale: &'a mut DeviceBuffer<f32>,
    pub len: u32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub beta1_correction: f32,
    pub beta2_correction: f32,
    pub eps: f32,
}

pub struct EmbeddingLookupGradArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub tokens: &'a DeviceBuffer<u32>,
    pub d_embedding_residual: &'a DeviceBuffer<f32>,
    pub d_token_embedding: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub embedding_dim: u32,
}
