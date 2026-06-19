use std::fs;
use std::path::PathBuf;

use hf_hub::HFClientSync;
use hf_hub::repository::RepoTreeEntry;

use crate::AppResult;
use crate::synth::{DATA_DIR, DATASET_NAME, DATASET_OWNER, PARQUET_DIR, PARQUET_FILE_PATTERN};

pub fn list_parquet_files() -> AppResult<Vec<String>> {
    let entries = HFClientSync::new()?
        .dataset(DATASET_OWNER, DATASET_NAME)
        .list_tree()
        .send()?;

    let mut files = entries
        .into_iter()
        .filter_map(|entry| match entry {
            RepoTreeEntry::File { path, .. } if is_synth_parquet(&path) => Some(path),
            _ => None,
        })
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

pub fn download_parquet_file(filename: &str) -> AppResult<PathBuf> {
    let local_dir = PathBuf::from(DATA_DIR).join(PARQUET_DIR);
    fs::create_dir_all(&local_dir)?;

    HFClientSync::new()?
        .dataset(DATASET_OWNER, DATASET_NAME)
        .download_file()
        .filename(filename)
        .local_dir(local_dir.clone())
        .send()?;

    Ok(local_dir.join(filename))
}

fn is_synth_parquet(path: &str) -> bool {
    let Some(prefix) = PARQUET_FILE_PATTERN.strip_suffix("*.parquet") else {
        return false;
    };
    path.starts_with(prefix) && path.ends_with(".parquet")
}
