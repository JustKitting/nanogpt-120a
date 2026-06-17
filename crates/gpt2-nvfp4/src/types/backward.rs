use cuda_core::DeviceBuffer;

use crate::GPT2_N_LAYER;

pub struct Gpt2BackwardContext<'a> {
    pub saved: Gpt2ForwardSaved<'a>,
    pub grads: Gpt2BackwardGrads<'a>,
}

pub struct Gpt2ForwardSaved<'a> {
    pub tokens: &'a DeviceBuffer<u32>,
    pub embedding_residual: &'a DeviceBuffer<f32>,
    pub blocks: [BlockForwardSaved<'a>; GPT2_N_LAYER],
    pub final_norm: LayerNormSaved<'a>,
    pub logits: &'a DeviceBuffer<f32>,
}

pub struct BlockForwardSaved<'a> {
    pub residual_in: &'a DeviceBuffer<f32>,
    pub ln_1: LayerNormSaved<'a>,
    pub qkv: &'a DeviceBuffer<f32>,
    pub attention_out: &'a DeviceBuffer<f32>,
    pub residual_after_attention: &'a DeviceBuffer<f32>,
    pub ln_2: LayerNormSaved<'a>,
    pub mlp_activation: &'a DeviceBuffer<f32>,
    pub residual_out: &'a DeviceBuffer<f32>,
}

pub struct LayerNormSaved<'a> {
    pub residual: &'a DeviceBuffer<f32>,
    pub normalized: &'a DeviceBuffer<f32>,
    pub normalized_amax: &'a DeviceBuffer<f32>,
}

pub struct Gpt2BackwardGrads<'a> {
    pub dlogits: &'a mut DeviceBuffer<f32>,
    pub d_embedding_residual: &'a mut DeviceBuffer<f32>,
    pub blocks: [BlockBackwardGrads<'a>; GPT2_N_LAYER],
    pub final_norm: LayerNormGrads<'a>,
}

pub struct BlockBackwardGrads<'a> {
    pub d_residual_in: &'a mut DeviceBuffer<f32>,
    pub ln_1: LayerNormGrads<'a>,
    pub d_qkv: &'a mut DeviceBuffer<f32>,
    pub d_attention_out: &'a mut DeviceBuffer<f32>,
    pub d_residual_after_attention: &'a mut DeviceBuffer<f32>,
    pub ln_2: LayerNormGrads<'a>,
    pub d_mlp_activation: &'a mut DeviceBuffer<f32>,
    pub d_residual_out: &'a mut DeviceBuffer<f32>,
}

pub struct LayerNormGrads<'a> {
    pub d_residual: &'a mut DeviceBuffer<f32>,
    pub d_normalized: &'a mut DeviceBuffer<f32>,
}
