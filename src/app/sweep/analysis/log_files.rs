use std::{fs, path::PathBuf};

use super::{super::history::Trial, logs::LogMetrics};

pub fn screen_path(trial: &Trial) -> PathBuf {
    if trial
        .log_path
        .file_name()
        .is_some_and(|name| name == "screen.log")
    {
        trial.log_path.clone()
    } else {
        trial.log_path.with_file_name("screen.log")
    }
}

pub fn full_path(trial: &Trial) -> Option<PathBuf> {
    (trial
        .log_path
        .file_name()
        .is_some_and(|name| name == "train.log"))
    .then(|| trial.log_path.clone())
}

pub fn read_log(path: impl Into<Option<PathBuf>>) -> Option<LogMetrics> {
    let path = path.into()?;
    let text = fs::read_to_string(path).ok()?;
    let mut metrics = LogMetrics::default();
    for line in text.lines() {
        metrics.saw_nan |= line.contains("loss=NaN") || line.contains("finite=false");
        metrics.panicked |= line.contains("panicked at") || line.contains("assertion failed");
        if line.starts_with("stopped_by_wall_clock=true") {
            metrics.elapsed_s = field(line, "elapsed_s=").and_then(parse_f64);
            metrics.completed_steps = field(line, "completed_steps=").and_then(parse_usize);
        }
        if line.starts_with("heldout_eval ") {
            metrics.val_loss = field(line, "val_loss=").and_then(parse_f64);
            metrics.elapsed_s = field(line, "train_elapsed_s=").and_then(parse_f64);
            metrics.completed_steps = field(line, "completed_steps=").and_then(parse_usize);
        }
    }
    Some(metrics)
}

fn field<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    let start = line.find(prefix)? + prefix.len();
    Some(line[start..].split_whitespace().next().unwrap_or(""))
}

fn parse_f64(value: &str) -> Option<f64> {
    value.parse::<f64>().ok().filter(|value| value.is_finite())
}

fn parse_usize(value: &str) -> Option<usize> {
    value.parse().ok()
}
