use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_TOKEN_ROWS, HiddenState, NextLatHiddenActivation, NextLatInputActivation};

pub struct NextLatBuffers {
    pub next_token_embeddings: DeviceBuffer<f32>,
    pub concat: DeviceBuffer<f32>,
    pub normalized: DeviceBuffer<f32>,
    pub normalized_amax: DeviceBuffer<f32>,
    pub mean: DeviceBuffer<f32>,
    pub inv_std: DeviceBuffer<f32>,
    pub input_bytes: DeviceBuffer<u8>,
    pub input_scales: DeviceBuffer<u8>,
    pub input_globals: DeviceBuffer<f32>,
    pub pre1: DeviceBuffer<f32>,
    pub act1: DeviceBuffer<f32>,
    pub act1_bytes: DeviceBuffer<u8>,
    pub act1_scales: DeviceBuffer<u8>,
    pub act1_globals: DeviceBuffer<f32>,
    pub pre2: DeviceBuffer<f32>,
    pub act2: DeviceBuffer<f32>,
    pub act2_bytes: DeviceBuffer<u8>,
    pub act2_scales: DeviceBuffer<u8>,
    pub act2_globals: DeviceBuffer<f32>,
    pub delta: DeviceBuffer<f32>,
    pub predicted: DeviceBuffer<f32>,
    pub losses: DeviceBuffer<f32>,
    pub d_predicted: DeviceBuffer<f32>,
}

impl NextLatBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            next_token_embeddings: zero(stream, HiddenState::LEN)?,
            concat: zero(stream, NextLatInputActivation::LEN)?,
            normalized: zero(stream, NextLatInputActivation::LEN)?,
            normalized_amax: zero(stream, GPT2_TOKEN_ROWS)?,
            mean: zero(stream, GPT2_TOKEN_ROWS)?,
            inv_std: zero(stream, GPT2_TOKEN_ROWS)?,
            input_bytes: DeviceBuffer::zeroed(stream, NextLatInputActivation::LEN / 2)?,
            input_scales: DeviceBuffer::zeroed(stream, NextLatInputActivation::LEN / 16)?,
            input_globals: zero(stream, GPT2_TOKEN_ROWS)?,
            pre1: zero(stream, NextLatHiddenActivation::LEN)?,
            act1: zero(stream, NextLatHiddenActivation::LEN)?,
            act1_bytes: DeviceBuffer::zeroed(stream, NextLatHiddenActivation::LEN / 2)?,
            act1_scales: DeviceBuffer::zeroed(stream, NextLatHiddenActivation::LEN / 16)?,
            act1_globals: zero(stream, GPT2_TOKEN_ROWS)?,
            pre2: zero(stream, NextLatHiddenActivation::LEN)?,
            act2: zero(stream, NextLatHiddenActivation::LEN)?,
            act2_bytes: DeviceBuffer::zeroed(stream, NextLatHiddenActivation::LEN / 2)?,
            act2_scales: DeviceBuffer::zeroed(stream, NextLatHiddenActivation::LEN / 16)?,
            act2_globals: zero(stream, GPT2_TOKEN_ROWS)?,
            delta: zero(stream, HiddenState::LEN)?,
            predicted: zero(stream, HiddenState::LEN)?,
            losses: zero(stream, GPT2_TOKEN_ROWS)?,
            d_predicted: zero(stream, HiddenState::LEN)?,
        })
    }
}

fn zero(stream: &CudaStream, len: usize) -> Result<DeviceBuffer<f32>, DriverError> {
    DeviceBuffer::zeroed(stream, len)
}
