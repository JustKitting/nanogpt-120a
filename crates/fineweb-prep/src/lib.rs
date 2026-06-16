use std::error::Error;

pub mod fineweb;
mod huggingface;
mod parquet_text;
mod pipeline;
mod shards;
mod tokenize;

pub type AppResult<T> = Result<T, Box<dyn Error>>;

pub use fineweb::{
    DATA_DIR, DATASET_CONFIG, DATASET_NAME, DATASET_OWNER, DATASET_REPO, DATASET_SPLIT,
    SHARD_FILE_PREFIX, SHARD_SIZE,
};
pub use pipeline::parse_data;
