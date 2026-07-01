use std::{fs, io, path::PathBuf};

use super::{candidate::Candidate, history::Trial};

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
        self.record.as_ref().map(Record::measured_trial)
    }

    pub fn promote_trial(&mut self, trial: &Trial, dry_run: bool) -> io::Result<bool> {
        if dry_run {
            return Ok(false);
        }

        let Some(record) = Record::from_trial(trial) else {
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
            .is_none_or(|record| val_loss < record.val_loss)
    }

    fn write(&self) -> io::Result<()> {
        self.record
            .as_ref()
            .map_or(Ok(()), |record| write::record(&self.path, record))
    }
}
