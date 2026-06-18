use std::fs;
use std::path::PathBuf;

use hf_hub::HFClientSync;

use crate::AppResult;
use crate::synth::{DATA_DIR, DATASET_NAME, DATASET_OWNER, PARQUET_DIR, PARQUET_FILE_PATTERN};

pub fn download_parquet_files() -> AppResult<Vec<PathBuf>> {
    let local_dir = PathBuf::from(DATA_DIR).join(PARQUET_DIR);
    HFClientSync::new()?
        .dataset(DATASET_OWNER, DATASET_NAME)
        .snapshot_download()
        .allow_patterns(vec![PARQUET_FILE_PATTERN.to_string()])
        .local_dir(local_dir.clone())
        .send()?;

    let mut files = fs::read_dir(local_dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    files.retain(|path| path.extension().is_some_and(|ext| ext == "parquet"));
    files.sort();
    Ok(files)
}
