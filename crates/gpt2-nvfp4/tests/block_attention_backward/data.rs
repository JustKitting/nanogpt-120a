use gpt2_nvfp4::{
    AttentionLogSumExp, GPT2_BATCH_SIZE, GPT2_N_HEAD, GPT2_SEQ_LEN, GPT2_TOKEN_ROWS, HiddenState,
};

pub const E2M1_MIN_PAIR: u8 = 0x11;
pub const E2M1_ONE_PAIR: u8 = 0x22;
pub const E4M3_ONE: u8 = 0x38;

pub fn hidden_values() -> Vec<f32> {
    (0..HiddenState::LEN)
        .map(|index| 0.000_122_070_31 * ((index % 19) as f32 + 1.0))
        .collect()
}

pub fn attention_log_sum_exp_values() -> Vec<f32> {
    let mut log_sum_exp = vec![0.0_f32; AttentionLogSumExp::LEN];
    for batch in 0..GPT2_BATCH_SIZE {
        for head in 0..GPT2_N_HEAD {
            for token in 0..GPT2_SEQ_LEN {
                let index = (batch * GPT2_N_HEAD + head) * GPT2_SEQ_LEN + token;
                log_sum_exp[index] = ((token + 1) as f32).ln();
            }
        }
    }
    log_sum_exp
}

pub fn row_global_scales() -> Vec<f32> {
    vec![1.0; GPT2_TOKEN_ROWS]
}

pub fn inv_std_values() -> Vec<f32> {
    vec![1.0; GPT2_TOKEN_ROWS]
}
