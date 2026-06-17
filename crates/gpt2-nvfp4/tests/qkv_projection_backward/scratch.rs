use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{AttentionQkvScratch, GPT2_N_EMBD, GPT2_QKV, GPT2_TOKEN_ROWS};
use rust_kernels_cuda::linear_backward::{LinearBackwardMsEdenScratch, MsEdenOperandScratch};

pub struct QkvBackwardScratch {
    pub error_t: DeviceBuffer<f32>,
    pub weight_t: DeviceBuffer<f32>,
    pub input_t: DeviceBuffer<f32>,
    e: OperandScratch,
    weight_t_h: OperandScratch,
    e_t: OperandScratch,
    input_t_h: OperandScratch,
}

impl QkvBackwardScratch {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            error_t: DeviceBuffer::<f32>::zeroed(stream, GPT2_QKV * GPT2_TOKEN_ROWS)?,
            weight_t: DeviceBuffer::<f32>::zeroed(stream, GPT2_QKV * GPT2_N_EMBD)?,
            input_t: DeviceBuffer::<f32>::zeroed(stream, GPT2_N_EMBD * GPT2_TOKEN_ROWS)?,
            e: OperandScratch::new(stream, GPT2_TOKEN_ROWS * GPT2_QKV, GPT2_TOKEN_ROWS)?,
            weight_t_h: OperandScratch::new(stream, GPT2_N_EMBD * GPT2_QKV, GPT2_N_EMBD)?,
            e_t: OperandScratch::new(stream, GPT2_QKV * GPT2_TOKEN_ROWS, GPT2_QKV)?,
            input_t_h: OperandScratch::new(stream, GPT2_N_EMBD * GPT2_TOKEN_ROWS, GPT2_N_EMBD)?,
        })
    }

    pub fn as_attention_scratch(&mut self) -> AttentionQkvScratch<'_> {
        AttentionQkvScratch {
            error_t: &mut self.error_t,
            weight_t: &mut self.weight_t,
            input_t: &mut self.input_t,
            linear: LinearBackwardMsEdenScratch {
                e_h: self.e.as_operand(),
                weight_t_h: self.weight_t_h.as_operand(),
                e_t_h: self.e_t.as_operand(),
                input_t_h: self.input_t_h.as_operand(),
            },
        }
    }
}

struct OperandScratch {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scales: DeviceBuffer<f32>,
    chunk_amax: DeviceBuffer<f32>,
    global_scale: DeviceBuffer<f32>,
}

impl OperandScratch {
    fn new(stream: &CudaStream, elements: usize, rows: usize) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::<u8>::zeroed(stream, elements / 2)?,
            scales: DeviceBuffer::<u8>::zeroed(stream, elements / 16)?,
            global_scales: DeviceBuffer::<f32>::zeroed(stream, rows)?,
            chunk_amax: DeviceBuffer::<f32>::zeroed(stream, elements / 32)?,
            global_scale: DeviceBuffer::<f32>::zeroed(stream, 1)?,
        })
    }

    fn as_operand(&mut self) -> MsEdenOperandScratch<'_> {
        MsEdenOperandScratch {
            bytes: &mut self.bytes,
            scales: &mut self.scales,
            global_scales: &mut self.global_scales,
            chunk_amax: &mut self.chunk_amax,
            global_scale: &mut self.global_scale,
        }
    }
}
