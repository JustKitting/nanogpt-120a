use gpt2_nvfp4::{
    GPT2_MLP, GPT2_N_EMBD, HiddenState, MlpDownWeightShape, MlpUpWeightShape, Nvfp4Shape,
};

use crate::common::nvfp4::repeating_identity_bytes;

pub fn normalized_input() -> Vec<f32> {
    hidden_values(|_, col| if col < GPT2_N_EMBD / 2 { 0.5 } else { -0.5 })
}

pub fn residual_input() -> Vec<f32> {
    hidden_values(|row, col| 0.125 + row as f32 * 0.000_244_140_62 + col as f32 * 1.0e-7)
}

fn hidden_values(value: impl Fn(usize, usize) -> f32) -> Vec<f32> {
    (0..HiddenState::LEN)
        .map(|i| value(i / GPT2_N_EMBD, i % GPT2_N_EMBD))
        .collect()
}

pub fn mlp_up_repeat_weight_bytes() -> Vec<u8> {
    repeating_identity_bytes(MlpUpWeightShape::BYTE_LEN, GPT2_MLP, GPT2_N_EMBD)
}

pub fn mlp_down_identity_weight_bytes() -> Vec<u8> {
    repeating_identity_bytes(MlpDownWeightShape::BYTE_LEN, GPT2_N_EMBD, GPT2_MLP)
}
