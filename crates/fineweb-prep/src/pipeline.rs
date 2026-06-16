use std::fs;

use gpt2_bpe::Gpt2Bpe;

use crate::fineweb::{DATA_DIR, PARQUET_DIR, SHARDS_DIR};
use crate::{AppResult, huggingface, parquet_text, shards};

pub fn parse_data() -> AppResult<()> {
    let files = huggingface::download_parquet_files()?;
    let data_dir = std::path::PathBuf::from(DATA_DIR);

    fs::create_dir_all(data_dir.join(PARQUET_DIR))?;
    fs::create_dir_all(data_dir.join(SHARDS_DIR))?;

    let tokenizer = Gpt2Bpe::from_default_assets()?;
    let mut writer = shards::ShardWriter::new();

    for path in &files {
        parquet_text::tokenize_parquet_file(path, &tokenizer, &mut writer)?;
    }

    writer.finish()
}
