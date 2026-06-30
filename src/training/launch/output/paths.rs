use std::path::PathBuf;

use crate::time_utils;

const RUNS_DIR: &str = "target/runs";

pub(super) fn default_run_dir(dataset: &str, label: &str) -> PathBuf {
    PathBuf::from(RUNS_DIR).join(format!(
        "{}_{}_{}",
        time_utils::utc_compact_stamp(),
        sanitize_path_part(dataset),
        sanitize_path_part(label)
    ))
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
