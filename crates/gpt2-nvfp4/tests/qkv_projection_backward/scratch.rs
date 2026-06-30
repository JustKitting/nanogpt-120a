use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{AttentionQkvScratch, GPT2_N_EMBD, GPT2_QKV, GPT2_TOKEN_ROWS};
use rust_kernels_cuda::linear_backward::LinearBackwardMsEdenScratchBuffers;

pub struct QkvBackwardScratch {
    pub error_t: DeviceBuffer<f32>,
    pub weight_t: DeviceBuffer<f32>,
    pub input_t: DeviceBuffer<f32>,
    linear: LinearBackwardMsEdenScratchBuffers,
}

impl QkvBackwardScratch {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            error_t: DeviceBuffer::<f32>::zeroed(stream, GPT2_QKV * GPT2_TOKEN_ROWS)?,
            weight_t: DeviceBuffer::<f32>::zeroed(stream, GPT2_QKV * GPT2_N_EMBD)?,
            input_t: DeviceBuffer::<f32>::zeroed(stream, GPT2_N_EMBD * GPT2_TOKEN_ROWS)?,
            linear: LinearBackwardMsEdenScratchBuffers::new(
                stream,
                GPT2_TOKEN_ROWS,
                GPT2_N_EMBD,
                GPT2_QKV,
            )?,
        })
    }

    pub fn as_attention_scratch(&mut self) -> AttentionQkvScratch<'_> {
        AttentionQkvScratch {
            error_t: &mut self.error_t,
            weight_t: &mut self.weight_t,
            input_t: &mut self.input_t,
            linear: self.linear.as_args(),
        }
    }
}
