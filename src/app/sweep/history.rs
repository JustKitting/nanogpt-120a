use std::{
    fs,
    path::{Path, PathBuf},
};

use super::candidate::Candidate;
use super::trial_row;

#[derive(Clone, Debug)]
pub struct Trial {
    pub candidate: Candidate,
    pub status: String,
    pub val_loss: Option<f64>,
    pub completed_steps: Option<usize>,
    pub log_path: PathBuf,
}

#[derive(Debug)]
pub struct History {
    pub trials: Vec<Trial>,
    path: PathBuf,
}

impl History {
    pub fn load(path: PathBuf) -> std::io::Result<Self> {
        let trials = trial_row::read_trials(&path);
        Ok(Self { trials, path })
    }

    pub fn append(&mut self, trial: Trial) -> std::io::Result<()> {
        trial_row::append(&self.path, &trial)?;
        self.trials.push(trial);
        Ok(())
    }

    pub fn append_unique(&mut self, trial: Trial) -> std::io::Result<bool> {
        if self
            .trials
            .iter()
            .any(|existing| existing.candidate.key() == trial.candidate.key())
        {
            return Ok(false);
        }
        self.append(trial)?;
        Ok(true)
    }
}

pub fn write_candidate(path: &Path, candidate: &Candidate) -> std::io::Result<()> {
    let mut text = String::new();
    for (name, value) in candidate.build_env().into_iter().chain(candidate.run_env()) {
        text.push_str(name);
        text.push('=');
        text.push_str(&value);
        text.push('\n');
    }
    fs::write(path, text)
}
