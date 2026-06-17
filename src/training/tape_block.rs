use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    AttentionLse, BlockForwardSaved, BlockForwardTape, GPT2_CONTEXT_LEN, HiddenState,
    MlpActivation, QkvActivation,
};

use super::tape_leaf::{LayerNormTapeBuffers, RowwiseTapeBuffers, zero};

pub struct BlockTapeBuffers {
    residual_in: DeviceBuffer<f32>,
    ln_1: LayerNormTapeBuffers,
    qkv_input: RowwiseTapeBuffers,
    qkv: DeviceBuffer<f32>,
    attention_out: DeviceBuffer<f32>,
    attention_lse: DeviceBuffer<f32>,
    c_proj_input: RowwiseTapeBuffers,
    residual_after_attention: DeviceBuffer<f32>,
    ln_2: LayerNormTapeBuffers,
    mlp_up_input: RowwiseTapeBuffers,
    mlp_up: DeviceBuffer<f32>,
    mlp_relu2: DeviceBuffer<f32>,
    mlp_down_input: RowwiseTapeBuffers,
    residual_out: DeviceBuffer<f32>,
}

impl BlockTapeBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            residual_in: zero(stream, HiddenState::LEN)?,
            ln_1: LayerNormTapeBuffers::new(stream)?,
            qkv_input: RowwiseTapeBuffers::new(stream, HiddenState::LEN, GPT2_CONTEXT_LEN)?,
            qkv: zero(stream, QkvActivation::LEN)?,
            attention_out: zero(stream, HiddenState::LEN)?,
            attention_lse: zero(stream, AttentionLse::LEN)?,
            c_proj_input: RowwiseTapeBuffers::new(stream, HiddenState::LEN, GPT2_CONTEXT_LEN)?,
            residual_after_attention: zero(stream, HiddenState::LEN)?,
            ln_2: LayerNormTapeBuffers::new(stream)?,
            mlp_up_input: RowwiseTapeBuffers::new(stream, HiddenState::LEN, GPT2_CONTEXT_LEN)?,
            mlp_up: zero(stream, MlpActivation::LEN)?,
            mlp_relu2: zero(stream, MlpActivation::LEN)?,
            mlp_down_input: RowwiseTapeBuffers::new(stream, MlpActivation::LEN, GPT2_CONTEXT_LEN)?,
            residual_out: zero(stream, HiddenState::LEN)?,
        })
    }

    pub fn tape(&mut self) -> BlockForwardTape<'_> {
        BlockForwardTape {
            residual_in: &mut self.residual_in,
            ln_1: self.ln_1.tape(),
            qkv_input_nvfp4: self.qkv_input.tape(),
            qkv: &mut self.qkv,
            attention_out: &mut self.attention_out,
            attention_lse: &mut self.attention_lse,
            c_proj_input_nvfp4: self.c_proj_input.tape(),
            residual_after_attention: &mut self.residual_after_attention,
            ln_2: self.ln_2.tape(),
            mlp_up_input_nvfp4: self.mlp_up_input.tape(),
            mlp_up: &mut self.mlp_up,
            mlp_relu2: &mut self.mlp_relu2,
            mlp_down_input_nvfp4: self.mlp_down_input.tape(),
            residual_out: &mut self.residual_out,
        }
    }

    pub fn saved(&self) -> BlockForwardSaved<'_> {
        BlockForwardSaved {
            residual_in: &self.residual_in,
            ln_1: self.ln_1.saved(),
            qkv_input_nvfp4: self.qkv_input.saved(),
            qkv: &self.qkv,
            attention_out: &self.attention_out,
            attention_lse: &self.attention_lse,
            c_proj_input_nvfp4: self.c_proj_input.saved(),
            residual_after_attention: &self.residual_after_attention,
            ln_2: self.ln_2.saved(),
            mlp_up_input_nvfp4: self.mlp_up_input.saved(),
            mlp_up: &self.mlp_up,
            mlp_relu2: &self.mlp_relu2,
            mlp_down_input_nvfp4: self.mlp_down_input.saved(),
            residual_out: &self.residual_out,
        }
    }
}
