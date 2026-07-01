use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    AttentionLogSumExp, HiddenState, Logits, MlpActivation, QkvActivation, GPT2_BATCH_SIZE,
    GPT2_N_HEAD, GPT2_SEQ_LEN, GPT2_TOKEN_ROWS,
};

use crate::common::forward_scratch::{CausalAttentionTcScratchBuffers, RowwiseNvfp4ScratchBuffers};

pub struct ForwardScratch {
    pub residual: DeviceBuffer<f32>,
    pub normalized: DeviceBuffer<f32>,
    pub normalized_amax: DeviceBuffer<f32>,
    pub mean: DeviceBuffer<f32>,
    pub inv_std: DeviceBuffer<f32>,
    pub hidden_nvfp4: RowwiseNvfp4ScratchBuffers,
    pub mlp_pre_activation: DeviceBuffer<f32>,
    pub mlp_activation: DeviceBuffer<f32>,
    pub mlp_activation_nvfp4: RowwiseNvfp4ScratchBuffers,
    pub qkv: DeviceBuffer<f32>,
    pub attention_log_sum_exp: DeviceBuffer<f32>,
    pub attention_tc: CausalAttentionTcScratchBuffers,
    pub logits: DeviceBuffer<f32>,
}

impl ForwardScratch {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            residual: DeviceBuffer::zeroed(stream, HiddenState::LEN)?,
            normalized: DeviceBuffer::zeroed(stream, HiddenState::LEN)?,
            normalized_amax: DeviceBuffer::zeroed(stream, GPT2_TOKEN_ROWS)?,
            mean: DeviceBuffer::zeroed(stream, GPT2_TOKEN_ROWS)?,
            inv_std: DeviceBuffer::zeroed(stream, GPT2_TOKEN_ROWS)?,
            hidden_nvfp4: RowwiseNvfp4ScratchBuffers::new(
                stream,
                HiddenState::LEN,
                GPT2_TOKEN_ROWS,
            )?,
            mlp_pre_activation: DeviceBuffer::zeroed(stream, MlpActivation::LEN)?,
            mlp_activation: DeviceBuffer::zeroed(stream, MlpActivation::LEN)?,
            mlp_activation_nvfp4: RowwiseNvfp4ScratchBuffers::new(
                stream,
                MlpActivation::LEN,
                GPT2_TOKEN_ROWS,
            )?,
            qkv: DeviceBuffer::zeroed(stream, QkvActivation::LEN)?,
            attention_log_sum_exp: DeviceBuffer::zeroed(stream, AttentionLogSumExp::LEN)?,
            attention_tc: CausalAttentionTcScratchBuffers::new(
                stream,
                HiddenState::LEN,
                GPT2_BATCH_SIZE,
                GPT2_N_HEAD,
                GPT2_SEQ_LEN,
            )?,
            logits: DeviceBuffer::zeroed(stream, Logits::LEN)?,
        })
    }
}
