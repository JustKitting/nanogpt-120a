use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    BlockBackwardGrads, GPT2_MLP, GPT2_N_EMBD, GPT2_QKV, HiddenState, LayerNormGrads,
    MlpActivation, QkvActivation,
};

use super::tape_leaf::zero;

pub struct BlockGradBuffers {
    pub(super) d_residual_in: DeviceBuffer<f32>,
    pub(super) ln_1: LayerNormGradBuffers,
    pub(super) d_qkv: DeviceBuffer<f32>,
    pub(super) d_attention_out: DeviceBuffer<f32>,
    pub(super) d_residual_after_attention: DeviceBuffer<f32>,
    pub(super) ln_2: LayerNormGradBuffers,
    pub(super) d_mlp_up: DeviceBuffer<f32>,
    pub(super) d_mlp_relu2: DeviceBuffer<f32>,
    pub(super) d_attn_qkv_weight: DeviceBuffer<f32>,
    pub(super) d_attn_c_proj_weight: DeviceBuffer<f32>,
    pub(super) d_mlp_c_fc_weight: DeviceBuffer<f32>,
    pub(super) d_mlp_c_proj_weight: DeviceBuffer<f32>,
    pub(super) d_residual_out: DeviceBuffer<f32>,
}

impl BlockGradBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            d_residual_in: zero(stream, HiddenState::LEN)?,
            ln_1: LayerNormGradBuffers::new(stream)?,
            d_qkv: zero(stream, QkvActivation::LEN)?,
            d_attention_out: zero(stream, HiddenState::LEN)?,
            d_residual_after_attention: zero(stream, HiddenState::LEN)?,
            ln_2: LayerNormGradBuffers::new(stream)?,
            d_mlp_up: zero(stream, MlpActivation::LEN)?,
            d_mlp_relu2: zero(stream, MlpActivation::LEN)?,
            d_attn_qkv_weight: zero(stream, GPT2_N_EMBD * GPT2_QKV)?,
            d_attn_c_proj_weight: zero(stream, GPT2_N_EMBD * GPT2_N_EMBD)?,
            d_mlp_c_fc_weight: zero(stream, GPT2_N_EMBD * GPT2_MLP)?,
            d_mlp_c_proj_weight: zero(stream, GPT2_MLP * GPT2_N_EMBD)?,
            d_residual_out: zero(stream, HiddenState::LEN)?,
        })
    }

    pub fn grads(&mut self) -> BlockBackwardGrads<'_> {
        BlockBackwardGrads {
            d_residual_in: &mut self.d_residual_in,
            ln_1: self.ln_1.grads(),
            d_qkv: &mut self.d_qkv,
            d_attention_out: &mut self.d_attention_out,
            d_residual_after_attention: &mut self.d_residual_after_attention,
            ln_2: self.ln_2.grads(),
            d_mlp_up: &mut self.d_mlp_up,
            d_mlp_relu2: &mut self.d_mlp_relu2,
            d_attn_qkv_weight: &mut self.d_attn_qkv_weight,
            d_attn_c_proj_weight: &mut self.d_attn_c_proj_weight,
            d_mlp_c_fc_weight: &mut self.d_mlp_c_fc_weight,
            d_mlp_c_proj_weight: &mut self.d_mlp_c_proj_weight,
            d_residual_out: &mut self.d_residual_out,
        }
    }
}

pub struct LayerNormGradBuffers {
    pub(super) d_residual: DeviceBuffer<f32>,
    pub(super) d_normalized: DeviceBuffer<f32>,
    pub(super) d_weight: DeviceBuffer<f32>,
    pub(super) d_bias: DeviceBuffer<f32>,
}

impl LayerNormGradBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            d_residual: zero(stream, HiddenState::LEN)?,
            d_normalized: zero(stream, HiddenState::LEN)?,
            d_weight: zero(stream, GPT2_N_EMBD)?,
            d_bias: zero(stream, GPT2_N_EMBD)?,
        })
    }

    pub fn grads(&mut self) -> LayerNormGrads<'_> {
        LayerNormGrads {
            d_residual: &mut self.d_residual,
            d_normalized: &mut self.d_normalized,
            d_weight: &mut self.d_weight,
            d_bias: &mut self.d_bias,
        }
    }
}
