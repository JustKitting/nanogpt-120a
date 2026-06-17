use gpt2_nvfp4::{
    GPT2_CONTEXT_LEN, GPT2_MLP, GPT2_N_EMBD, HiddenState, MlpDownWeightShape, MlpUpWeightShape,
    Nvfp4Shape,
};

pub fn normalized_input() -> Vec<f32> {
    let mut normalized = vec![0.0_f32; HiddenState::LEN];
    for row in 0..GPT2_CONTEXT_LEN {
        let row_base = row * GPT2_N_EMBD;
        normalized[row_base..row_base + GPT2_N_EMBD / 2].fill(0.5);
        normalized[row_base + GPT2_N_EMBD / 2..row_base + GPT2_N_EMBD].fill(-0.5);
    }
    normalized
}

pub fn residual_input() -> Vec<f32> {
    let mut residual = vec![0.0_f32; HiddenState::LEN];
    for row in 0..GPT2_CONTEXT_LEN {
        let row_base = row * GPT2_N_EMBD;
        for col in 0..GPT2_N_EMBD {
            residual[row_base + col] = 0.125 + row as f32 * 0.000_244_140_62 + col as f32 * 1.0e-6;
        }
    }
    residual
}

pub fn mlp_up_repeat_weight_bytes() -> Vec<u8> {
    let mut bytes = vec![0_u8; MlpUpWeightShape::BYTE_LEN];
    for col in 0..GPT2_MLP {
        set_e2m1_one(&mut bytes, col * GPT2_N_EMBD + col % GPT2_N_EMBD);
    }
    bytes
}

pub fn mlp_down_identity_weight_bytes() -> Vec<u8> {
    let mut bytes = vec![0_u8; MlpDownWeightShape::BYTE_LEN];
    for col in 0..GPT2_N_EMBD {
        set_e2m1_one(&mut bytes, col * GPT2_MLP + col);
    }
    bytes
}

fn set_e2m1_one(bytes: &mut [u8], element: usize) {
    let byte = &mut bytes[element / 2];
    if element & 1 == 0 {
        *byte = (*byte & 0xf0) | 0x2;
    } else {
        *byte = (*byte & 0x0f) | 0x20;
    }
}
