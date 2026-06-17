use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    AttentionLogSumExp, GPT2_TOKEN_ROWS, HiddenState, Logits, MlpActivation, QkvActivation,
};

use super::grads::BackwardBuffers;
use super::optimizer::OptimizerScratch;
use super::optimizer_state::OptimizerStateBuffers;
use super::optimizer_tc_scratch::AuroraScratchBuffers;
use super::scratch::BackwardScratchBuffers;
use super::tape::ForwardTapeBuffers;

pub struct TrainBuffers {
    pub residual: DeviceBuffer<f32>,
    pub normalized: DeviceBuffer<f32>,
    pub normalized_amax: DeviceBuffer<f32>,
    pub mean: DeviceBuffer<f32>,
    pub inv_std: DeviceBuffer<f32>,
    pub hidden_bytes: DeviceBuffer<u8>,
    pub hidden_scales: DeviceBuffer<u8>,
    pub hidden_globals: DeviceBuffer<f32>,
    pub mlp_pre: DeviceBuffer<f32>,
    pub mlp_act: DeviceBuffer<f32>,
    pub mlp_bytes: DeviceBuffer<u8>,
    pub mlp_scales: DeviceBuffer<u8>,
    pub mlp_globals: DeviceBuffer<f32>,
    pub qkv: DeviceBuffer<f32>,
    pub log_sum_exp: DeviceBuffer<f32>,
    pub logits: DeviceBuffer<f32>,
    pub tape: ForwardTapeBuffers,
    pub backward: BackwardBuffers,
    pub scratch: BackwardScratchBuffers,
    pub optimizer: OptimizerScratch,
    pub optimizer_state: OptimizerStateBuffers,
    pub aurora: AuroraScratchBuffers,
}

impl TrainBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            residual: zero(stream, HiddenState::LEN)?,
            normalized: zero(stream, HiddenState::LEN)?,
            normalized_amax: zero(stream, GPT2_TOKEN_ROWS)?,
            mean: zero(stream, GPT2_TOKEN_ROWS)?,
            inv_std: zero(stream, GPT2_TOKEN_ROWS)?,
            hidden_bytes: DeviceBuffer::zeroed(stream, HiddenState::LEN / 2)?,
            hidden_scales: DeviceBuffer::zeroed(stream, HiddenState::LEN / 16)?,
            hidden_globals: zero(stream, GPT2_TOKEN_ROWS)?,
            mlp_pre: zero(stream, MlpActivation::LEN)?,
            mlp_act: zero(stream, MlpActivation::LEN)?,
            mlp_bytes: DeviceBuffer::zeroed(stream, MlpActivation::LEN / 2)?,
            mlp_scales: DeviceBuffer::zeroed(stream, MlpActivation::LEN / 16)?,
            mlp_globals: zero(stream, GPT2_TOKEN_ROWS)?,
            qkv: zero(stream, QkvActivation::LEN)?,
            log_sum_exp: zero(stream, AttentionLogSumExp::LEN)?,
            logits: zero(stream, Logits::LEN)?,
            tape: ForwardTapeBuffers::new(stream)?,
            backward: BackwardBuffers::new(stream)?,
            scratch: BackwardScratchBuffers::new(stream)?,
            optimizer: OptimizerScratch::new(stream)?,
            optimizer_state: OptimizerStateBuffers::new(stream)?,
            aurora: AuroraScratchBuffers::new(stream)?,
        })
    }
}

fn zero(stream: &CudaStream, len: usize) -> Result<DeviceBuffer<f32>, DriverError> {
    DeviceBuffer::zeroed(stream, len)
}
