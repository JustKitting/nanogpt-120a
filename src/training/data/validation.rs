use std::path::Path;

use gpt2_nvfp4::GPT2_SEQ_LEN;

use crate::AppResult;

pub(in crate::training) const VALIDATION_WINDOWS: usize = 4;

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

    Ok(tokens[start..start + needed].to_vec())
}

pub(super) fn train_end(token_count: usize) -> usize {
    let validation_tokens = VALIDATION_WINDOWS * (GPT2_SEQ_LEN + 1);
    token_count
        .saturating_sub(validation_tokens)
        .max(GPT2_SEQ_LEN + 1)
}
