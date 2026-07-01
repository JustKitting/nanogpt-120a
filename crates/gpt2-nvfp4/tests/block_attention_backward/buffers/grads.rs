use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    BlockBackwardGrads, HiddenState, LayerNormGrads, MlpActivation, QkvActivation, GPT2_MLP,
    GPT2_N_EMBD, GPT2_QKV,
};

use crate::data;

pub struct GradBuffers {
    pub d_residual_in: DeviceBuffer<f32>,
    pub d_qkv: DeviceBuffer<f32>,
    pub d_attention_out: DeviceBuffer<f32>,
    pub d_attn_qkv_weight: DeviceBuffer<f32>,
    pub d_attn_qkv_bias: DeviceBuffer<f32>,
    pub d_attn_c_proj_weight: DeviceBuffer<f32>,
    pub d_attn_c_proj_bias: DeviceBuffer<f32>,
    d_residual_after_attention: DeviceBuffer<f32>,
    ln1: LayerNormGradBuffers,
    ln2: LayerNormGradBuffers,
    d_mlp_up: DeviceBuffer<f32>,
    d_mlp_relu2: DeviceBuffer<f32>,
    d_mlp_c_fc_weight: DeviceBuffer<f32>,
    d_mlp_c_fc_bias: DeviceBuffer<f32>,
    d_mlp_c_proj_weight: DeviceBuffer<f32>,
    d_mlp_c_proj_bias: DeviceBuffer<f32>,
    d_residual_out: DeviceBuffer<f32>,
}

impl GradBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            d_residual_in: DeviceBuffer::zeroed(stream, HiddenState::LEN)?,
            d_qkv: DeviceBuffer::zeroed(stream, QkvActivation::LEN)?,
            d_attention_out: DeviceBuffer::zeroed(stream, HiddenState::LEN)?,
            d_attn_qkv_weight: DeviceBuffer::zeroed(stream, GPT2_N_EMBD * GPT2_QKV)?,
            d_attn_qkv_bias: DeviceBuffer::zeroed(stream, GPT2_QKV)?,
            d_attn_c_proj_weight: DeviceBuffer::zeroed(stream, GPT2_N_EMBD * GPT2_N_EMBD)?,
            d_attn_c_proj_bias: DeviceBuffer::zeroed(stream, GPT2_N_EMBD)?,
            d_residual_after_attention: DeviceBuffer::from_host(stream, &data::hidden_values())?,
            ln1: LayerNormGradBuffers::new(stream)?,
            ln2: LayerNormGradBuffers::new(stream)?,
            d_mlp_up: DeviceBuffer::zeroed(stream, MlpActivation::LEN)?,
            d_mlp_relu2: DeviceBuffer::zeroed(stream, MlpActivation::LEN)?,
            d_mlp_c_fc_weight: DeviceBuffer::zeroed(stream, GPT2_N_EMBD * GPT2_MLP)?,
            d_mlp_c_fc_bias: DeviceBuffer::zeroed(stream, GPT2_MLP)?,
            d_mlp_c_proj_weight: DeviceBuffer::zeroed(stream, GPT2_MLP * GPT2_N_EMBD)?,
            d_mlp_c_proj_bias: DeviceBuffer::zeroed(stream, GPT2_N_EMBD)?,
            d_residual_out: DeviceBuffer::zeroed(stream, HiddenState::LEN)?,
        })
    }

    pub fn block(&mut self) -> BlockBackwardGrads<'_> {
        BlockBackwardGrads {
            d_residual_in: &mut self.d_residual_in,
            ln_1: self.ln1.grads(),
            d_qkv: &mut self.d_qkv,
            d_attention_out: &mut self.d_attention_out,
            d_residual_after_attention: &mut self.d_residual_after_attention,
            ln_2: self.ln2.grads(),
            d_mlp_up: &mut self.d_mlp_up,
            d_mlp_relu2: &mut self.d_mlp_relu2,
            d_attn_qkv_weight: &mut self.d_attn_qkv_weight,
            d_attn_qkv_bias: &mut self.d_attn_qkv_bias,
            d_attn_c_proj_weight: &mut self.d_attn_c_proj_weight,
            d_attn_c_proj_bias: &mut self.d_attn_c_proj_bias,
            d_mlp_c_fc_weight: &mut self.d_mlp_c_fc_weight,
            d_mlp_c_fc_bias: &mut self.d_mlp_c_fc_bias,
            d_mlp_c_proj_weight: &mut self.d_mlp_c_proj_weight,
            d_mlp_c_proj_bias: &mut self.d_mlp_c_proj_bias,
            d_residual_out: &mut self.d_residual_out,
        }
    }
}

struct LayerNormGradBuffers {
    d_residual: DeviceBuffer<f32>,
    d_normalized: DeviceBuffer<f32>,
    d_weight: DeviceBuffer<f32>,
    d_bias: DeviceBuffer<f32>,
}

impl LayerNormGradBuffers {
    fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            d_residual: DeviceBuffer::zeroed(stream, HiddenState::LEN)?,
            d_normalized: DeviceBuffer::zeroed(stream, HiddenState::LEN)?,
            d_weight: DeviceBuffer::zeroed(stream, GPT2_N_EMBD)?,
            d_bias: DeviceBuffer::zeroed(stream, GPT2_N_EMBD)?,
        })
    }

    fn grads(&mut self) -> LayerNormGrads<'_> {
        LayerNormGrads {
            d_residual: &mut self.d_residual,
            d_normalized: &mut self.d_normalized,
            d_weight: &mut self.d_weight,
            d_bias: &mut self.d_bias,
        }
    }
}
