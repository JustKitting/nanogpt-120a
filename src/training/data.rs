use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};
use llama2_tokenizer::Llama2Tokenizer;
use synth_prep::synth::SHARDS_DIR;
use synth_prep::{DATA_DIR, SHARD_FILE_PREFIX};

use crate::AppResult;

const TRAIN_DATASET_ENV: &str = "TRAIN_DATASET";
const TRAIN_REPEAT_BATCH_ENV: &str = "TRAIN_REPEAT_BATCH";
const DATASET_SYNTH: &str = "synth";
const DATASET_SHAKESPEARE: &str = "shakespeare";
const SHAKESPEARE_URL: &str =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
const SHAKESPEARE_DIR: &str = "data/shakespeare";
const SHAKESPEARE_RAW: &str = "input.txt";
const SHAKESPEARE_SHARD: &str = "shakespeare_llama2_train_000000.bin";
const VALIDATION_WINDOWS: usize = 4;

pub struct TokenDataLoader {
    path: PathBuf,
    tokens: Vec<u16>,
    validation_tokens: Option<Vec<u16>>,
    validation_path: Option<PathBuf>,
    offset: usize,
    train_end: usize,
    repeat_first_window: bool,
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
        ensure_synth_shards()?;
        let validation_path = first_val_shard()?;
        let validation_tokens = read_u16_tokens(&validation_path)?;
        Self::from_path(
            first_train_shard()?,
            Some((validation_path, validation_tokens)),
        )
    }

    pub fn from_shakespeare() -> AppResult<Self> {
        let path = ensure_shakespeare_shard()?;
        Self::from_path(path, None)
    }

    pub fn next_batch(&mut self) -> AppResult<TokenWindowBatch> {
        let len = GPT2_SEQ_LEN + 1;
        if self.train_end < len {
            return Err(format!("{} has fewer than {len} tokens", self.path.display()).into());
        }

        let mut offsets = Vec::with_capacity(GPT2_BATCH_SIZE);
        if self.repeat_first_window {
            offsets.resize(GPT2_BATCH_SIZE, 0);
        } else {
            for _ in 0..GPT2_BATCH_SIZE {
                if self.offset + len > self.train_end {
                    self.offset = 0;
                }
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
        self.tokens.len()
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

    fn from_path(path: PathBuf, validation: Option<(PathBuf, Vec<u16>)>) -> AppResult<Self> {
        let tokens = read_u16_tokens(&path)?;
        let train_end = if validation.is_some() {
            tokens.len()
        } else {
            train_end(tokens.len())
        };
        let (validation_path, validation_tokens) = validation
            .map(|(path, tokens)| (Some(path), Some(tokens)))
            .unwrap_or((None, None));
        Ok(Self {
            path,
            tokens,
            validation_tokens,
            validation_path,
            offset: 0,
            train_end,
            repeat_first_window: repeat_first_window(),
        })
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

fn training_dataset() -> String {
    std::env::var(TRAIN_DATASET_ENV).unwrap_or_else(|_| DATASET_SYNTH.to_string())
}

fn repeat_first_window() -> bool {
    std::env::var(TRAIN_REPEAT_BATCH_ENV)
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

fn first_train_shard() -> AppResult<PathBuf> {
    first_shard("train")
}

fn first_val_shard() -> AppResult<PathBuf> {
    first_shard("val")
}

fn first_shard(split: &str) -> AppResult<PathBuf> {
    let dir = Path::new(DATA_DIR).join(SHARDS_DIR);
    let prefix = format!("{SHARD_FILE_PREFIX}_{split}_");
    let mut shards = Vec::new();

    if !dir.exists() {
        return Err(format!("{} does not exist after SYNTH prep", dir.display()).into());
    }

    for entry in fs::read_dir(&dir)? {
        let path = entry?.path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.starts_with(&prefix) && file_name.ends_with(".bin") {
            shards.push(path);
        }
    }

    shards.sort();
    shards
        .into_iter()
        .next()
        .ok_or_else(|| format!("no {split} shards found in {}", dir.display()).into())
}

fn ensure_synth_shards() -> AppResult<()> {
    if first_train_shard().is_ok() && first_val_shard().is_ok() {
        return Ok(());
    }

    synth_prep::parse_data()?;
    first_train_shard()?;
    first_val_shard().map(|_| ())
}

fn ensure_shakespeare_shard() -> AppResult<PathBuf> {
    let dir = Path::new(SHAKESPEARE_DIR);
    let shard_dir = dir.join(SHARDS_DIR);
    let shard_path = shard_dir.join(SHAKESPEARE_SHARD);
    if shard_path.exists() {
        return Ok(shard_path);
    }

    fs::create_dir_all(&shard_dir)?;
    let raw_path = dir.join(SHAKESPEARE_RAW);
    if !raw_path.exists() {
        let text = reqwest::blocking::get(SHAKESPEARE_URL)?
            .error_for_status()?
            .text()?;
        fs::create_dir_all(dir)?;
        fs::write(&raw_path, text)?;
    }

    let text = fs::read_to_string(raw_path)?;
    let tokenizer = Llama2Tokenizer::from_default_assets()?;
    let mut tokens = Vec::new();
    tokens.push(u16::try_from(tokenizer.bos_token())?);
    for id in tokenizer.encode_ordinary(&text)? {
        tokens.push(u16::try_from(id)?);
    }
    write_u16_tokens(&shard_path, &tokens)?;
    Ok(shard_path)
}

fn read_u16_tokens(path: &Path) -> AppResult<Vec<u16>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 2 != 0 {
        return Err(format!("{} has odd byte length", path.display()).into());
    }

    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn write_u16_tokens(path: &Path, tokens: &[u16]) -> AppResult<()> {
    let mut bytes = Vec::with_capacity(tokens.len() * 2);
    for &token in tokens {
        bytes.extend_from_slice(&token.to_ne_bytes());
    }

    fs::write(path, bytes).map_err(|err| {
        io::Error::new(
            err.kind(),
            format!("failed to write {}: {err}", path.display()),
        )
        .into()
    })
}
