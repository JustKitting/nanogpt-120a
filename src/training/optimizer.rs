use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_N_EMBD, GPT2_VOCAB_SIZE};

pub struct OptimizerScratch {
    pub(super) fp32_workspace: DeviceBuffer<f32>,
    pub(super) amax: DeviceBuffer<f32>,
    pub(super) next_global_scale: DeviceBuffer<f32>,
}

impl OptimizerScratch {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            fp32_workspace: DeviceBuffer::zeroed(stream, GPT2_VOCAB_SIZE * GPT2_N_EMBD)?,
            amax: DeviceBuffer::zeroed(stream, 1)?,
            next_global_scale: DeviceBuffer::zeroed(stream, 1)?,
        })
    }
}
