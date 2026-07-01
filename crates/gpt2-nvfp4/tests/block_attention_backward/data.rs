use gpt2_nvfp4::HiddenState;

pub fn hidden_values() -> Vec<f32> {
    (0..HiddenState::LEN)
        .map(|index| 0.000_122_070_31 * ((index % 19) as f32 + 1.0))
        .collect()
}
