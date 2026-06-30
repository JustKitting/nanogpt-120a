use std::path::{Path, PathBuf};

use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};

use crate::AppResult;

mod shakespeare;
mod synth;
mod tokens;

const TRAIN_DATASET_ENV: &str = "TRAIN_DATASET";
const TRAIN_REPEAT_BATCH_ENV: &str = "TRAIN_REPEAT_BATCH";
const DATASET_SYNTH: &str = "synth";
const DATASET_SHAKESPEARE: &str = "shakespeare";
const VALIDATION_WINDOWS: usize = 4;

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
    pub fn training_dataset_name() -> String {
        training_dataset()
    }

    pub fn validation_window_count() -> usize {
        VALIDATION_WINDOWS
    }

    pub fn from_training_dataset() -> AppResult<Self> {
        match training_dataset().as_str() {
            DATASET_SYNTH => Self::from_synth(),
            DATASET_SHAKESPEARE => Self::from_shakespeare(),
            dataset => Err(format!(
                "unknown TRAIN_DATASET={dataset}; expected synth or shakespeare"
            )
            .into()),
        }
    }

    pub fn from_synth() -> AppResult<Self> {
        synth::ensure_shards()?;
        let train_paths = synth::train_shards()?;
        let validation_path = synth::first_val_shard()?;
        let validation_tokens = tokens::read_u16_tokens(&validation_path)?;
        Self::from_train_paths(
            train_paths,
            Some((validation_path, validation_tokens)),
            false,
            false,
        )
    }

    pub fn from_shakespeare() -> AppResult<Self> {
        let path = shakespeare::ensure_shard()?;
        Self::from_train_paths(vec![path], None, true, true)
    }

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

    fn from_train_paths(
        train_paths: Vec<PathBuf>,
        validation: Option<(PathBuf, Vec<u16>)>,
        wrap_train: bool,
        reserve_validation_tail: bool,
    ) -> AppResult<Self> {
        let path = train_paths
            .first()
            .cloned()
            .ok_or("training dataset has no train shards")?;
        let tokens = tokens::read_u16_tokens(&path)?;
        let train_end = if validation.is_some() {
            tokens.len()
        } else if reserve_validation_tail {
            train_end(tokens.len())
        } else {
            tokens.len()
        };
        let total_train_tokens = token_count_paths(&train_paths)?;
        let (validation_path, validation_tokens) = validation
            .map(|(path, tokens)| (Some(path), Some(tokens)))
            .unwrap_or((None, None));
        Ok(Self {
            path,
            train_paths,
            train_path_index: 0,
            tokens,
            validation_tokens,
            validation_path,
            offset: 0,
            train_end,
            total_train_tokens,
            repeat_first_window: repeat_first_window(),
            wrap_train,
        })
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

fn validation_windows(path: &Path, tokens: &[u16], start: usize) -> AppResult<Vec<u16>> {
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

fn train_end(token_count: usize) -> usize {
    let validation_tokens = VALIDATION_WINDOWS * (GPT2_SEQ_LEN + 1);
    token_count
        .saturating_sub(validation_tokens)
        .max(GPT2_SEQ_LEN + 1)
}

fn token_count_paths(paths: &[PathBuf]) -> AppResult<usize> {
    let mut total = 0usize;
    for path in paths {
        let bytes = path.metadata()?.len();
        if bytes % 2 != 0 {
            return Err(format!("{} has odd byte length", path.display()).into());
        }
        total += (bytes / 2) as usize;
    }
    Ok(total)
}

fn training_dataset() -> String {
    std::env::var(TRAIN_DATASET_ENV).unwrap_or_else(|_| DATASET_SYNTH.to_string())
}

fn repeat_first_window() -> bool {
    std::env::var(TRAIN_REPEAT_BATCH_ENV)
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}
