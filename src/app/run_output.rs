use std::fs;
use std::path::{Path, PathBuf};

use time::OffsetDateTime;

use crate::AppResult;

const TRAIN_RUN_DIR_ENV: &str = "TRAIN_RUN_DIR";
const RUNS_DIR: &str = "target/runs";

pub(crate) struct RunOutput {
    dir: PathBuf,
}

impl RunOutput {
    pub fn new(dataset: &str, steps: usize) -> AppResult<Self> {
        let dir = std::env::var(TRAIN_RUN_DIR_ENV)
            .ok()
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| default_run_dir(dataset, steps));
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn path(&self, file_name: &str) -> PathBuf {
        self.dir.join(file_name)
    }

    pub fn write_info(&self, info: &str) -> AppResult {
        fs::write(self.path("run_info.txt"), info)?;
        Ok(())
    }
}

pub(crate) fn ensure_parent(path: &Path) -> AppResult {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn default_run_dir(dataset: &str, steps: usize) -> PathBuf {
    PathBuf::from(RUNS_DIR).join(format!(
        "{}_{}_{}steps",
        utc_stamp(),
        sanitize_path_part(dataset),
        steps
    ))
}

fn utc_stamp() -> String {
    let now = OffsetDateTime::now_utc();
    format!(
        "{:04}{:02}{:02}_{:02}{:02}{:02}Z",
        now.year(),
        now.month() as u8,
        now.day(),
        now.hour(),
        now.minute(),
        now.second()
    )
}

fn sanitize_path_part(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}
