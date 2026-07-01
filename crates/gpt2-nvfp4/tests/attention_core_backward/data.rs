use gpt2_nvfp4::HiddenState;

pub fn d_out_values() -> Vec<f32> {
    (0..HiddenState::LEN)
        .map(|index| (index % 17) as f32 * 0.000_244_140_63)
        .collect()
}
