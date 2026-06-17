use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    AttentionCProjScratch, BlockAttentionBackwardScratch, GPT2_N_EMBD, GPT2_QKV, GPT2_TOKEN_ROWS,
};
use rust_kernels_cuda::linear_backward::{LinearBackwardMsEdenScratch, MsEdenOperandScratch};

use super::attention_core_scratch::AttentionCoreScratchBuffers;

pub struct BlockAttentionScratch {
    c_proj: LinearScratch,
    qkv: LinearScratch,
    core: AttentionCoreScratchBuffers,
}

impl BlockAttentionScratch {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            c_proj: LinearScratch::new(stream, GPT2_N_EMBD, GPT2_N_EMBD)?,
            qkv: LinearScratch::new(stream, GPT2_N_EMBD, GPT2_QKV)?,
            core: AttentionCoreScratchBuffers::new(stream)?,
        })
    }

    pub fn block(&mut self) -> BlockAttentionBackwardScratch<'_> {
        BlockAttentionBackwardScratch {
            c_proj: self.c_proj.as_attention_scratch(),
            core: self.core.args(),
            qkv: self.qkv.as_attention_scratch(),
        }
    }
}

struct LinearScratch {
    error_t: DeviceBuffer<f32>,
    weight_t: DeviceBuffer<f32>,
    input_t: DeviceBuffer<f32>,
    e: OperandScratch,
    weight_t_h: OperandScratch,
    e_t: OperandScratch,
    input_t_h: OperandScratch,
}

impl LinearScratch {
    fn new(stream: &CudaStream, input_dim: usize, output_dim: usize) -> Result<Self, DriverError> {
        Ok(Self {
            error_t: DeviceBuffer::zeroed(stream, output_dim * GPT2_TOKEN_ROWS)?,
            weight_t: DeviceBuffer::zeroed(stream, output_dim * input_dim)?,
            input_t: DeviceBuffer::zeroed(stream, input_dim * GPT2_TOKEN_ROWS)?,
            e: OperandScratch::new(stream, GPT2_TOKEN_ROWS * output_dim, GPT2_TOKEN_ROWS)?,
            weight_t_h: OperandScratch::new(stream, input_dim * output_dim, input_dim)?,
            e_t: OperandScratch::new(stream, output_dim * GPT2_TOKEN_ROWS, output_dim)?,
            input_t_h: OperandScratch::new(stream, input_dim * GPT2_TOKEN_ROWS, input_dim)?,
        })
    }

    fn as_attention_scratch(&mut self) -> AttentionCProjScratch<'_> {
        AttentionCProjScratch {
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
            bytes: DeviceBuffer::zeroed(stream, elements / 2)?,
            scales: DeviceBuffer::zeroed(stream, elements / 16)?,
            global_scales: DeviceBuffer::zeroed(stream, rows)?,
            chunk_amax: DeviceBuffer::zeroed(stream, elements / 32)?,
            global_scale: DeviceBuffer::zeroed(stream, 1)?,
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
