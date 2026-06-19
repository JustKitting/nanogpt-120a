use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};
use llama2_tokenizer::Llama2Tokenizer;
use synth_prep::synth::SHARDS_DIR;
use synth_prep::{DATA_DIR, DEFAULT_TRAIN_SHARD_COUNT, SHARD_FILE_PREFIX, SHARD_SIZE};

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
        ensure_synth_shards()?;
        let train_paths = train_shards()?;
        let validation_path = first_val_shard()?;
        let validation_tokens = read_u16_tokens(&validation_path)?;
        Self::from_train_paths(
            train_paths,
            Some((validation_path, validation_tokens)),
            false,
            false,
        )
    }

    pub fn from_shakespeare() -> AppResult<Self> {
        let path = ensure_shakespeare_shard()?;
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
        let tokens = read_u16_tokens(&path)?;
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
        self.tokens = read_u16_tokens(&self.path)?;
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

fn train_shards() -> AppResult<Vec<PathBuf>> {
    let shards = shards_for_split("train")?
        .into_iter()
        .filter(|path| is_full_synth_shard(path))
        .collect::<Vec<_>>();
    if shards.is_empty() {
        return Err("no full SYNTH train shards found".into());
    }
    Ok(shards)
}

fn first_val_shard() -> AppResult<PathBuf> {
    shards_for_split("val")?
        .into_iter()
        .next()
        .ok_or_else(|| format!("no val shards found in {}", synth_shard_dir().display()).into())
}

fn shards_for_split(split: &str) -> AppResult<Vec<PathBuf>> {
    let dir = synth_shard_dir();
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
    Ok(shards)
}

fn synth_shard_dir() -> PathBuf {
    Path::new(DATA_DIR).join(SHARDS_DIR)
}

fn is_full_synth_shard(path: &Path) -> bool {
    path.metadata()
        .is_ok_and(|metadata| metadata.len() == (SHARD_SIZE * 2) as u64)
}

fn ensure_synth_shards() -> AppResult<()> {
    if train_shards().is_ok_and(|shards| shards.len() >= DEFAULT_TRAIN_SHARD_COUNT)
        && first_val_shard().is_ok()
    {
        return Ok(());
    }

    synth_prep::parse_data_for_train_shards(DEFAULT_TRAIN_SHARD_COUNT)?;
    let train_shard_count = train_shards()?.len();
    if train_shard_count < DEFAULT_TRAIN_SHARD_COUNT {
        return Err(format!(
            "expected {DEFAULT_TRAIN_SHARD_COUNT} full SYNTH train shards, found {train_shard_count}"
        )
        .into());
    }
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
