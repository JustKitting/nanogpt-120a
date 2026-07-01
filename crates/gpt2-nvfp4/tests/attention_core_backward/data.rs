use gpt2_nvfp4::{
    AttentionLogSumExp, GPT2_BATCH_SIZE, GPT2_N_HEAD, GPT2_SEQ_LEN, HiddenState,
};

pub fn d_out_values() -> Vec<f32> {
    (0..HiddenState::LEN)
        .map(|index| (index % 17) as f32 * 0.000_244_140_63)
        .collect()
}

pub fn log_sum_exp_values() -> Vec<f32> {
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
