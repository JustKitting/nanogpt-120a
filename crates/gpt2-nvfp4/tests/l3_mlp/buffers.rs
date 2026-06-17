use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_CONTEXT_LEN, HiddenState, MlpActivation};

pub struct ScratchBuffers {
    pub residual: DeviceBuffer<f32>,
    pub normalized: DeviceBuffer<f32>,
    pub amax: DeviceBuffer<f32>,
    pub input_bytes: DeviceBuffer<u8>,
    pub input_scales: DeviceBuffer<u8>,
    pub input_global_scales: DeviceBuffer<f32>,
    pub activation: DeviceBuffer<f32>,
    pub activation_bytes: DeviceBuffer<u8>,
    pub activation_scales: DeviceBuffer<u8>,
    pub activation_global_scales: DeviceBuffer<f32>,
}

impl ScratchBuffers {
    pub fn new(
        stream: &CudaStream,
        normalized: &[f32],
        amax: &[f32],
        residual: &[f32],
    ) -> Result<Self, DriverError> {
        Ok(Self {
            residual: DeviceBuffer::from_host(stream, residual)?,
            normalized: DeviceBuffer::from_host(stream, normalized)?,
            amax: DeviceBuffer::from_host(stream, amax)?,
            input_bytes: DeviceBuffer::zeroed(stream, HiddenState::LEN / 2)?,
            input_scales: DeviceBuffer::zeroed(stream, HiddenState::LEN / 16)?,
            input_global_scales: DeviceBuffer::zeroed(stream, GPT2_CONTEXT_LEN)?,
            activation: DeviceBuffer::zeroed(stream, MlpActivation::LEN)?,
            activation_bytes: DeviceBuffer::zeroed(stream, MlpActivation::LEN / 2)?,
            activation_scales: DeviceBuffer::zeroed(stream, MlpActivation::LEN / 16)?,
            activation_global_scales: DeviceBuffer::zeroed(stream, GPT2_CONTEXT_LEN)?,
        })
    }
}
