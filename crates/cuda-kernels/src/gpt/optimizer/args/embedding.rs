use cuda_core::{CudaStream, DeviceBuffer};

pub struct EmbeddingLookupGradArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub tokens: &'a DeviceBuffer<u32>,
    pub d_embedding_residual: &'a DeviceBuffer<f32>,
    pub d_token_embedding: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub embedding_dim: u32,
}
