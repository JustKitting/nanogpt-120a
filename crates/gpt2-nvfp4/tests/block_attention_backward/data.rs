use gpt2_nvfp4::HiddenState;

pub const E2M1_MIN_PAIR: u8 = 0x11;
pub const E2M1_ONE_PAIR: u8 = 0x22;
pub const E4M3_ONE: u8 = 0x38;

pub fn hidden_values() -> Vec<f32> {
    (0..HiddenState::LEN)
        .map(|index| 0.000_122_070_31 * ((index % 19) as f32 + 1.0))
        .collect()
}
