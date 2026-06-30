use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

use super::history::Trial;
use crate::fs_utils::ensure_parent;

mod format;
mod parse;
#[cfg(test)]
mod tests;

use format::{format_trial, header};
use parse::parse_trial;

pub fn read_trials(path: &Path) -> Vec<Trial> {
    fs::read_to_string(path)
        .ok()
        .map(|text| text.lines().skip(1).filter_map(parse_trial).collect())
        .unwrap_or_default()
}

pub fn append(path: &Path, trial: &Trial) -> std::io::Result<()> {
    ensure_parent(path)?;
    let new_file = !path.exists();
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if new_file {
        writeln!(file, "{}", header())?;
    }
    writeln!(file, "{}", format_trial(trial))
}
