use std::fs;
use std::path::PathBuf;

use hf_hub::HFClientSync;

use crate::AppResult;
use crate::fineweb::{
    DATA_DIR, DATASET_CONFIG, DATASET_NAME, DATASET_OWNER, DATASET_SPLIT, PARQUET_DIR,
};

pub fn download_parquet_files() -> AppResult<Vec<PathBuf>> {
    let local_dir = PathBuf::from(DATA_DIR).join(PARQUET_DIR);
    HFClientSync::new()?
        .dataset(DATASET_OWNER, DATASET_NAME)
        .snapshot_download()
        .allow_patterns(vec![format!("{DATASET_CONFIG}/{DATASET_SPLIT}/*.parquet")])
        .local_dir(local_dir.clone())
        .send()?;

    let parquet_dir = local_dir.join(DATASET_CONFIG).join(DATASET_SPLIT);
    let mut files = fs::read_dir(parquet_dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    files.retain(|path| path.extension().is_some_and(|ext| ext == "parquet"));
    files.sort();
    Ok(files)
}
