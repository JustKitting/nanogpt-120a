use std::fs;
use std::path::{Path, PathBuf};

use fineweb_prep::fineweb::SHARDS_DIR;
use fineweb_prep::{DATA_DIR, SHARD_FILE_PREFIX};
use gpt2_nvfp4::GPT2_CONTEXT_LEN;

use crate::AppResult;

const TRAIN_SHARD_ENV: &str = "TRAIN_SHARD";

pub struct TokenDataLoader {
    path: PathBuf,
    tokens: Vec<u16>,
    offset: usize,
}

pub struct TokenWindow<'a> {
    pub tokens: &'a [u16],
    pub source: &'a Path,
    pub offset: usize,
}

impl TokenDataLoader {
    pub fn from_env_or_default() -> AppResult<Self> {
        let path = match std::env::var_os(TRAIN_SHARD_ENV) {
            Some(path) => PathBuf::from(path),
            None => first_train_shard()?,
        };
        Self::from_path(path)
    }

    pub fn next_window(&mut self) -> AppResult<TokenWindow<'_>> {
        let len = GPT2_CONTEXT_LEN + 1;
        if self.tokens.len() < len {
            return Err(format!("{} has fewer than {len} tokens", self.path.display()).into());
        }
        if self.offset + len > self.tokens.len() {
            self.offset = 0;
        }

        let offset = self.offset;
        self.offset += GPT2_CONTEXT_LEN;

        Ok(TokenWindow {
            tokens: &self.tokens[offset..offset + len],
            source: &self.path,
            offset,
        })
    }

    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    fn from_path(path: PathBuf) -> AppResult<Self> {
        Ok(Self {
            tokens: read_u16_tokens(&path)?,
            path,
            offset: 0,
        })
    }
}

fn first_train_shard() -> AppResult<PathBuf> {
    let dir = Path::new(DATA_DIR).join(SHARDS_DIR);
    let prefix = format!("{SHARD_FILE_PREFIX}_train_");
    let mut shards = Vec::new();

    if !dir.exists() {
        return Err(format!(
            "{} does not exist; run fineweb_prep::parse_data() first or set TRAIN_SHARD",
            dir.display()
        )
        .into());
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
        .ok_or_else(|| format!("no train shards found in {}", dir.display()).into())
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
