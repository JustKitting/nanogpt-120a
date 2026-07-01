use gpt2_nvfp4::{
    HiddenState, Nvfp4Shape, QkvWeightShape, ResidualWeightShape, GPT2_CONTEXT_LEN, GPT2_N_EMBD,
    GPT2_QKV,
};

use crate::common::nvfp4::repeating_identity_bytes;

pub fn hidden_input() -> (Vec<f32>, Vec<f32>) {
    let mut hidden = vec![0.0_f32; HiddenState::LEN];
    let mut amax = vec![0.0_f32; GPT2_CONTEXT_LEN];
    for (row, (chunk, amax)) in hidden.chunks_mut(GPT2_N_EMBD).zip(&mut amax).enumerate() {
        let value = 0.125 + (row % 7) as f32 * 0.0625;
        chunk.fill(value);
        *amax = value;
    }
    (hidden, amax)
}

pub fn residual_input() -> Vec<f32> {
    let mut residual = vec![0.0_f32; HiddenState::LEN];
    for (row, chunk) in residual.chunks_mut(GPT2_N_EMBD).enumerate() {
        chunk.fill(0.25 + row as f32 * 0.000_976_562_5);
    }
    residual
}

pub fn qkv_identity_weight_bytes() -> Vec<u8> {
    repeating_identity_bytes(QkvWeightShape::BYTE_LEN, GPT2_QKV, GPT2_N_EMBD)
}

pub fn c_proj_identity_weight_bytes() -> Vec<u8> {
    repeating_identity_bytes(ResidualWeightShape::BYTE_LEN, GPT2_N_EMBD, GPT2_N_EMBD)
}
