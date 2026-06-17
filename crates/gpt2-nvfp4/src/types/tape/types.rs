use cuda_core::DeviceBuffer;

use crate::GPT2_N_LAYER;

pub struct Gpt2ForwardTape<'a> {
    pub embedding_residual: &'a mut DeviceBuffer<f32>,
    pub blocks: [BlockForwardTape<'a>; GPT2_N_LAYER],
    pub final_norm: LayerNormTape<'a>,
    pub lm_head_input_nvfp4: RowwiseNvfp4Tape<'a>,
    pub logits: &'a mut DeviceBuffer<f32>,
}

pub struct BlockForwardTape<'a> {
    pub residual_in: &'a mut DeviceBuffer<f32>,
    pub ln_1: LayerNormTape<'a>,
    pub qkv_input_nvfp4: RowwiseNvfp4Tape<'a>,
    pub qkv: &'a mut DeviceBuffer<f32>,
    pub attention_out: &'a mut DeviceBuffer<f32>,
    pub c_proj_input_nvfp4: RowwiseNvfp4Tape<'a>,
    pub residual_after_attention: &'a mut DeviceBuffer<f32>,
    pub ln_2: LayerNormTape<'a>,
    pub mlp_up_input_nvfp4: RowwiseNvfp4Tape<'a>,
    pub mlp_up: &'a mut DeviceBuffer<f32>,
    pub mlp_relu2: &'a mut DeviceBuffer<f32>,
    pub mlp_down_input_nvfp4: RowwiseNvfp4Tape<'a>,
    pub residual_out: &'a mut DeviceBuffer<f32>,
}

pub struct LayerNormTape<'a> {
    pub residual: &'a mut DeviceBuffer<f32>,
    pub normalized: &'a mut DeviceBuffer<f32>,
    pub mean: &'a mut DeviceBuffer<f32>,
    pub inv_std: &'a mut DeviceBuffer<f32>,
}

pub struct RowwiseNvfp4Tape<'a> {
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scales: &'a mut DeviceBuffer<f32>,
}
