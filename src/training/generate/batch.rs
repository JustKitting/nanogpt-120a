use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};

use crate::AppResult;

pub(super) fn generation_batch(tokens: &[u32], pad_token: u32) -> AppResult<(Vec<u16>, u32)> {
    let context_len = tokens.len().min(GPT2_SEQ_LEN);
    let context_start = tokens.len() - context_len;
    let row = context_len.saturating_sub(1) as u32;
    let window_len = GPT2_SEQ_LEN + 1;
    let pad = u16::try_from(pad_token)?;
    let mut one = vec![pad; window_len];

    for (dst, &token) in one.iter_mut().zip(tokens[context_start..].iter()) {
        *dst = u16::try_from(token)?;
    }

    let mut windows = Vec::with_capacity(GPT2_BATCH_SIZE * window_len);
    for _ in 0..GPT2_BATCH_SIZE {
        windows.extend_from_slice(&one);
    }

    Ok((windows, row))
}
