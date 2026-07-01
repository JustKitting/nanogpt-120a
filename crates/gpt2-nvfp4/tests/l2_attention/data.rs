use gpt2_nvfp4::{
    GPT2_CONTEXT_LEN, GPT2_N_EMBD, GPT2_QKV, HiddenState, Nvfp4Shape, QkvWeightShape,
    ResidualWeightShape,
};

use super::nvfp4_common::repeating_identity_bytes;

pub fn hidden_input() -> (Vec<f32>, Vec<f32>) {
    let mut hidden = vec![0.0_f32; HiddenState::LEN];
    let mut amax = vec![0.0_f32; GPT2_CONTEXT_LEN];
    for row in 0..GPT2_CONTEXT_LEN {
        let value = 0.125 + (row % 7) as f32 * 0.0625;
        hidden[row * GPT2_N_EMBD..(row + 1) * GPT2_N_EMBD].fill(value);
        amax[row] = value;
    }
    (hidden, amax)
}

pub fn residual_input() -> Vec<f32> {
    let mut residual = vec![0.0_f32; HiddenState::LEN];
    for row in 0..GPT2_CONTEXT_LEN {
        residual[row * GPT2_N_EMBD..(row + 1) * GPT2_N_EMBD]
            .fill(0.25 + row as f32 * 0.000_976_562_5);
    }
    residual
}

pub fn qkv_identity_weight_bytes() -> Vec<u8> {
    repeating_identity_bytes(QkvWeightShape::BYTE_LEN, GPT2_QKV, GPT2_N_EMBD)
}

pub fn c_proj_identity_weight_bytes() -> Vec<u8> {
    repeating_identity_bytes(ResidualWeightShape::BYTE_LEN, GPT2_N_EMBD, GPT2_N_EMBD)
}
