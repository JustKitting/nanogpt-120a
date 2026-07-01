use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{HiddenState, MlpActivation, GPT2_CONTEXT_LEN};

use crate::common::forward_scratch::RowwiseNvfp4ScratchBuffers;

pub struct ScratchBuffers {
    pub residual: DeviceBuffer<f32>,
    pub normalized: DeviceBuffer<f32>,
    pub amax: DeviceBuffer<f32>,
    pub mean: DeviceBuffer<f32>,
    pub inv_std: DeviceBuffer<f32>,
    pub input_nvfp4: RowwiseNvfp4ScratchBuffers,
    pub pre_activation: DeviceBuffer<f32>,
    pub activation: DeviceBuffer<f32>,
    pub activation_nvfp4: RowwiseNvfp4ScratchBuffers,
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
            mean: DeviceBuffer::zeroed(stream, GPT2_CONTEXT_LEN)?,
            inv_std: DeviceBuffer::zeroed(stream, GPT2_CONTEXT_LEN)?,
            input_nvfp4: RowwiseNvfp4ScratchBuffers::new(
                stream,
                HiddenState::LEN,
                GPT2_CONTEXT_LEN,
            )?,
            pre_activation: DeviceBuffer::zeroed(stream, MlpActivation::LEN)?,
            activation: DeviceBuffer::zeroed(stream, MlpActivation::LEN)?,
            activation_nvfp4: RowwiseNvfp4ScratchBuffers::new(
                stream,
                MlpActivation::LEN,
                GPT2_CONTEXT_LEN,
            )?,
        })
    }
}
