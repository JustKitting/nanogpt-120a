use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NextLatShape {
    pub row_count: u32,
    pub embedding_dim: u32,
    pub seq_len: u32,
    pub batch_size: u32,
    pub lambda: f32,
}

unsafe impl DeviceCopy for NextLatShape {}

pub struct NextLatConcatArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub next_token_embeddings: &'a DeviceBuffer<f32>,
    pub current_states: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
}

pub struct NextLatSmoothL1Args<'a, 'out> {
    pub stream: &'a CudaStream,
    pub predicted_next_states: &'a DeviceBuffer<f32>,
    pub target_states: &'a DeviceBuffer<f32>,
    pub losses: &'out mut DeviceBuffer<f32>,
    pub d_predicted_next_states: &'out mut DeviceBuffer<f32>,
    pub batch_size: u32,
    pub seq_len: u32,
    pub embedding_dim: u32,
    pub lambda: f32,
}
