use std::fs;

use llama2_tokenizer::Llama2Tokenizer;

use crate::synth::{DATA_DIR, PARQUET_DIR, SHARDS_DIR};
use crate::{AppResult, huggingface, parquet_text, shards};

pub fn parse_data() -> AppResult<()> {
    let files = huggingface::list_parquet_files()?;
    if files.is_empty() {
        return Err("PleIAs/SYNTH did not list any synth_*.parquet files".into());
    }
    let data_dir = std::path::PathBuf::from(DATA_DIR);

    fs::create_dir_all(data_dir.join(PARQUET_DIR))?;
    fs::create_dir_all(data_dir.join(SHARDS_DIR))?;

    let tokenizer = Llama2Tokenizer::from_default_assets()?;
    let mut writer = shards::ShardWriter::new();

    for file in &files {
        let path = huggingface::download_parquet_file(file)?;
        parquet_text::tokenize_parquet_file(&path, &tokenizer, &mut writer)?;
        if writer.has_default_train_and_val_shards() {
            break;
        }
    }

    writer.finish()
}
