use std::{fs, io, path::PathBuf};

use super::{
    candidate::{Candidate, MIN_N_LAYER},
    history::Trial,
};

mod parse;
mod record;
mod write;

use record::Record;

pub struct Baseline {
    path: PathBuf,
    record: Option<Record>,
}

impl Baseline {
    pub fn load(path: PathBuf) -> io::Result<Self> {
        let record = fs::read_to_string(&path)
            .ok()
            .and_then(|text| parse::record(&text));
        Ok(Self { path, record })
    }

    pub fn candidate(&self) -> Option<&Candidate> {
        self.record.as_ref().map(|record| &record.candidate)
    }

    pub fn val_loss(&self) -> Option<f64> {
        self.record.as_ref().map(|record| record.val_loss)
    }

    pub fn measured_trial(&self) -> Option<Trial> {
        let record = self.record.as_ref()?;
        Some(Trial {
            candidate: record.candidate.clone(),
            status: "success".to_string(),
            val_loss: Some(record.val_loss),
            completed_steps: record.completed_steps,
            elapsed_s: record.elapsed_s,
            screen_val_loss: record.screen_loss,
            screen_completed_steps: record.screen_completed_steps,
            screen_elapsed_s: record.screen_elapsed_s,
            screen_reason: record.screen_reason.clone(),
            log_path: record.log_path.clone(),
        })
    }

    pub fn promote_trial(&mut self, trial: &Trial, dry_run: bool) -> io::Result<bool> {
        if dry_run {
            return Ok(false);
        }

        let Some(record) = trial_record(trial) else {
            return Ok(false);
        };

        if !self.is_improvement(record.val_loss) {
            return Ok(false);
        }

        self.record = Some(record);
        self.write()?;
        Ok(true)
    }

    fn is_improvement(&self, val_loss: f64) -> bool {
        self.record
            .as_ref()
            .map(|record| val_loss < record.val_loss)
            .unwrap_or(true)
    }

    fn write(&self) -> io::Result<()> {
        let Some(record) = &self.record else {
            return Ok(());
        };
        write::record(&self.path, record)
    }
}

fn trial_record(trial: &Trial) -> Option<Record> {
    if trial.status != "success" {
        return None;
    }
    if trial.candidate.n_layer < MIN_N_LAYER {
        return None;
    }
    Some(Record {
        val_loss: trial.val_loss.filter(|loss| loss.is_finite())?,
        completed_steps: trial.completed_steps,
        elapsed_s: trial.elapsed_s,
        screen_loss: trial.screen_val_loss,
        screen_completed_steps: trial.screen_completed_steps,
        screen_elapsed_s: trial.screen_elapsed_s,
        screen_reason: trial.screen_reason.clone(),
        log_path: trial.log_path.clone(),
        candidate: trial.candidate.clone(),
    })
}
