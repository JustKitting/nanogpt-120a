use gpt2_nvfp4::{
    AttentionBackwardSeeds, GPT2_TOKEN_ROWS, Gpt2Rng, HiddenState, Nvfp4Shape, QkvActivation,
    QkvWeightShape,
};

const E2M1_MIN_PAIR: u8 = 0x11;
const E4M3_ONE: u8 = 0x38;

pub fn qkv_input_bytes() -> Vec<u8> {
    vec![E2M1_MIN_PAIR; HiddenState::LEN / 2]
}

pub fn hidden_scales() -> Vec<u8> {
    vec![E4M3_ONE; HiddenState::LEN / 16]
}

pub fn row_global_scales() -> Vec<f32> {
    vec![1.0; GPT2_TOKEN_ROWS]
}

pub fn qkv_weight_bytes() -> Vec<u8> {
    vec![E2M1_MIN_PAIR; QkvWeightShape::BYTE_LEN]
}

pub fn d_qkv_values() -> Vec<f32> {
    (0..QkvActivation::LEN)
        .map(|index| 0.000_244_140_63 * ((index % 11) as f32 + 1.0))
        .collect()
}

pub fn seeds() -> AttentionBackwardSeeds {
    let mut rng = Gpt2Rng::new(0x5156_4b56);
    AttentionBackwardSeeds::from_rng(&mut rng)
}
