use std::{fs, path::PathBuf};

use super::super::history::Trial;
use super::super::parse::RunResult;
use super::logs::LogMetrics;

pub fn screen_path(trial: &Trial) -> PathBuf {
    if has_log_file(trial, "screen.log") {
        trial.log_path.clone()
    } else {
        trial.log_path.with_file_name("screen.log")
    }
}

pub fn full_path(trial: &Trial) -> Option<PathBuf> {
    has_log_file(trial, "train.log").then(|| trial.log_path.clone())
}

fn has_log_file(trial: &Trial, file_name: &str) -> bool {
    trial
        .log_path
        .file_name()
        .is_some_and(|name| name == file_name)
}

pub fn read_log(path: impl Into<Option<PathBuf>>) -> Option<LogMetrics> {
    let path = path.into()?;
    let text = fs::read_to_string(path).ok()?;
    let mut result = RunResult::default();
    let mut panicked = false;
    for line in text.lines() {
        result.update(line);
        panicked |= line.contains("panicked at") || line.contains("assertion failed");
    }
    Some(LogMetrics {
        val_loss: result.val_loss,
        elapsed_s: result.last_elapsed_s,
        saw_nan: result.saw_nan,
        panicked,
    })
}
