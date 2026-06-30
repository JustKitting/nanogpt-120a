use std::path::Path;

use gpt2_nvfp4::GPT2_SEQ_LEN;

use crate::AppResult;

pub(super) const VALIDATION_WINDOWS: usize = 4;

pub(super) fn validation_windows(path: &Path, tokens: &[u16], start: usize) -> AppResult<Vec<u16>> {
    let len = GPT2_SEQ_LEN + 1;
    if tokens.len() < len {
        return Err(format!("{} has fewer than {len} tokens", path.display()).into());
    }

    let needed = VALIDATION_WINDOWS * len;
    if tokens.len() < start + needed {
        return Err(format!(
            "{} has fewer than {} validation tokens",
            path.display(),
            start + needed
        )
        .into());
    }

    let mut validation_tokens = Vec::with_capacity(needed);
    for batch in 0..VALIDATION_WINDOWS {
        let offset = start + batch * len;
        validation_tokens.extend_from_slice(&tokens[offset..offset + len]);
    }
    Ok(validation_tokens)
}

pub(super) fn train_end(token_count: usize) -> usize {
    let validation_tokens = VALIDATION_WINDOWS * (GPT2_SEQ_LEN + 1);
    token_count
        .saturating_sub(validation_tokens)
        .max(GPT2_SEQ_LEN + 1)
}
