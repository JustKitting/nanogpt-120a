use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use fineweb_prep::fineweb::SHARDS_DIR;
use fineweb_prep::{DATA_DIR, SHARD_FILE_PREFIX};
use gpt2_bpe::Gpt2Bpe;
use gpt2_nvfp4::GPT2_CONTEXT_LEN;

use crate::AppResult;

const TRAIN_DATASET_ENV: &str = "TRAIN_DATASET";
const DATASET_FINEWEB: &str = "fineweb";
const DATASET_SHAKESPEARE: &str = "shakespeare";
const SHAKESPEARE_URL: &str =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
const SHAKESPEARE_DIR: &str = "data/shakespeare";
const SHAKESPEARE_RAW: &str = "input.txt";
const SHAKESPEARE_SHARD: &str = "shakespeare_train_000000.bin";

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
    pub fn from_training_dataset() -> AppResult<Self> {
        match training_dataset()?.as_str() {
            DATASET_FINEWEB => Self::from_fineweb(),
            DATASET_SHAKESPEARE => Self::from_shakespeare(),
            dataset => Err(format!(
                "unknown TRAIN_DATASET={dataset}; expected fineweb or shakespeare"
            )
            .into()),
        }
    }

    pub fn from_fineweb() -> AppResult<Self> {
        ensure_fineweb_shard()?;
        Self::from_path(first_train_shard()?)
    }

    pub fn from_shakespeare() -> AppResult<Self> {
        let path = ensure_shakespeare_shard()?;
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

fn training_dataset() -> AppResult<String> {
    Ok(std::env::var(TRAIN_DATASET_ENV).unwrap_or_else(|_| DATASET_FINEWEB.to_string()))
}

fn first_train_shard() -> AppResult<PathBuf> {
    let dir = Path::new(DATA_DIR).join(SHARDS_DIR);
    let prefix = format!("{SHARD_FILE_PREFIX}_train_");
    let mut shards = Vec::new();

    if !dir.exists() {
        return Err(format!("{} does not exist after FineWeb prep", dir.display()).into());
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

fn ensure_fineweb_shard() -> AppResult<()> {
    if first_train_shard().is_ok() {
        return Ok(());
    }

    fineweb_prep::parse_data()?;
    first_train_shard().map(|_| ())
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
    let tokenizer = Gpt2Bpe::from_default_assets()?;
    let mut tokens = Vec::new();
    tokens.push(u16::try_from(tokenizer.eot_token())?);
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
