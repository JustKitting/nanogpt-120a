use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::AppResult;

pub(crate) fn load_bpe_ranks(
    path: impl AsRef<Path>,
) -> AppResult<HashMap<(String, String), usize>> {
    let vocab = fs::read_to_string(path)?;
    let mut ranks = HashMap::new();

    for (line_index, line) in vocab.lines().enumerate() {
        if line_index == 0 && line.starts_with("#version:") {
            continue;
        }

        let Some((left, right)) = line.split_once(' ') else {
            continue;
        };
        ranks.insert((left.to_owned(), right.to_owned()), ranks.len());
    }

    Ok(ranks)
}

pub(crate) fn build_decoder(encoder: &HashMap<String, u32>) -> Vec<String> {
    let max_id = encoder.values().copied().max().unwrap_or(0) as usize;
    let mut decoder = vec![String::new(); max_id + 1];

    for (token, id) in encoder {
        decoder[*id as usize] = token.clone();
    }

    decoder
}
