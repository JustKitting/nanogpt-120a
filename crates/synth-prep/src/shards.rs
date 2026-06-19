use std::path::PathBuf;

use bytemuck::cast_slice;

use crate::AppResult;
use crate::synth::{DATA_DIR, SHARD_FILE_PREFIX, SHARD_SIZE, SHARDS_DIR};

pub struct ShardWriter {
    output_dir: PathBuf,
    shard_index: usize,
    target_train_shards: usize,
    tokens: Vec<u16>,
}

impl ShardWriter {
    pub fn new(target_train_shards: usize) -> Self {
        Self {
            output_dir: PathBuf::from(DATA_DIR).join(SHARDS_DIR),
            shard_index: 0,
            target_train_shards,
            tokens: Vec::with_capacity(SHARD_SIZE),
        }
    }

    pub fn push(&mut self, token: u16) -> AppResult<()> {
        self.tokens.push(token);
        if self.tokens.len() == SHARD_SIZE {
            self.flush_current()?;
        }
        Ok(())
    }

    pub fn has_required_train_and_val_shards(&self) -> bool {
        self.shard_index > self.target_train_shards
    }

    pub fn finish(mut self) -> AppResult<()> {
        if !self.has_required_train_and_val_shards() && !self.tokens.is_empty() {
            self.flush_current()?;
        }
        Ok(())
    }

    fn flush_current(&mut self) -> AppResult<()> {
        let split = if self.shard_index == 0 {
            "val"
        } else {
            "train"
        };
        let path = self.output_dir.join(format!(
            "{SHARD_FILE_PREFIX}_{split}_{:06}.bin",
            self.shard_index
        ));

        std::fs::write(path, cast_slice(&self.tokens))?;
        self.tokens.clear();
        self.shard_index += 1;
        Ok(())
    }
}
