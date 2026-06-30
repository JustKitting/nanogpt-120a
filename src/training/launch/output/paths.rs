use std::{
    fs,
    path::{Path, PathBuf},
};

use time::OffsetDateTime;

use crate::AppResult;

const RUNS_DIR: &str = "target/runs";

pub(in crate::training::launch) fn ensure_parent(path: &Path) -> AppResult {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

pub(super) fn default_run_dir(dataset: &str, label: &str) -> PathBuf {
    PathBuf::from(RUNS_DIR).join(format!(
        "{}_{}_{}",
        utc_stamp(),
        sanitize_path_part(dataset),
        sanitize_path_part(label)
    ))
}

fn utc_stamp() -> String {
    let now = OffsetDateTime::now_utc();
    format!(
        "{:04}{:02}{:02}_{:02}{:02}{:02}Z",
        now.year(),
        u8::from(now.month()),
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
