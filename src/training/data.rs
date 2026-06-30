use std::path::PathBuf;

use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};

use crate::AppResult;

mod factory;
mod shakespeare;
mod source;
mod synth;
mod tokens;
mod validation;

use validation::{VALIDATION_WINDOWS, train_end, validation_windows};

pub struct TokenDataLoader {
    path: PathBuf,
    train_paths: Vec<PathBuf>,
    train_path_index: usize,
    tokens: Vec<u16>,
    validation_tokens: Option<Vec<u16>>,
    validation_path: Option<PathBuf>,
    offset: usize,
    train_end: usize,
    total_train_tokens: usize,
    repeat_first_window: bool,
    wrap_train: bool,
}

pub struct TokenWindowBatch {
    pub tokens: Vec<u16>,
    pub source: PathBuf,
    pub offset: usize,
    pub batch_size: usize,
    pub seq_len: usize,
}

impl TokenDataLoader {
    pub fn next_batch(&mut self) -> AppResult<TokenWindowBatch> {
        let len = GPT2_SEQ_LEN + 1;
        let batch_span = GPT2_BATCH_SIZE * GPT2_SEQ_LEN + 1;
        if self.train_end < len {
            return Err(format!("{} has fewer than {len} tokens", self.path.display()).into());
        }
        if !self.repeat_first_window && self.offset + batch_span > self.train_end {
            self.advance_train_shard()?;
        }

        let mut offsets = Vec::with_capacity(GPT2_BATCH_SIZE);
        if self.repeat_first_window {
            offsets.resize(GPT2_BATCH_SIZE, 0);
        } else {
            for _ in 0..GPT2_BATCH_SIZE {
                offsets.push(self.offset);
                self.offset += GPT2_SEQ_LEN;
            }
        }

        let mut tokens = Vec::with_capacity(GPT2_BATCH_SIZE * len);
        for &offset in &offsets {
            tokens.extend_from_slice(&self.tokens[offset..offset + len]);
        }
        Ok(TokenWindowBatch {
            tokens,
            source: self.path.clone(),
            offset: offsets.first().copied().unwrap_or(0),
            batch_size: GPT2_BATCH_SIZE,
            seq_len: GPT2_SEQ_LEN,
        })
    }

    pub fn token_count(&self) -> usize {
        self.total_train_tokens
    }

    pub fn validation_tokens(&self) -> AppResult<Vec<u16>> {
        if let Some(tokens) = &self.validation_tokens {
            let path = self.validation_path.as_deref().unwrap_or(&self.path);
            return validation_windows(path, tokens, 0);
        }

        let needed = VALIDATION_WINDOWS * (GPT2_SEQ_LEN + 1);
        let start = self.tokens.len().saturating_sub(needed);
        validation_windows(&self.path, &self.tokens, start)
    }

    fn advance_train_shard(&mut self) -> AppResult<()> {
        let next_index = self.train_path_index + 1;
        if next_index >= self.train_paths.len() {
            if !self.wrap_train {
                return Err(format!(
                    "ran out of fresh train shards after {}; prepare more SYNTH train shards",
                    self.path.display()
                )
                .into());
            }
            self.train_path_index = 0;
        } else {
            self.train_path_index = next_index;
        }

        self.path = self.train_paths[self.train_path_index].clone();
        self.tokens = tokens::read_u16_tokens(&self.path)?;
        self.train_end = if self.validation_tokens.is_some() {
            self.tokens.len()
        } else {
            train_end(self.tokens.len())
        };
        self.offset = 0;
        Ok(())
    }
}
