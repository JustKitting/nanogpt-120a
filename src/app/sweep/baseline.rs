use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

use super::{
    candidate::{Candidate, MIN_N_LAYER},
    history::Trial,
};

const DEFAULT_SEQ_LEN: usize = 1024;

pub struct Baseline {
    path: PathBuf,
    record: Option<Record>,
}

#[derive(Clone)]
struct Record {
    val_loss: f64,
    completed_steps: Option<usize>,
    elapsed_s: Option<f64>,
    screen_loss: Option<f64>,
    screen_completed_steps: Option<usize>,
    screen_elapsed_s: Option<f64>,
    screen_reason: Option<String>,
    log_path: PathBuf,
    candidate: Candidate,
}

impl Baseline {
    pub fn load(path: PathBuf) -> io::Result<Self> {
        let record = fs::read_to_string(&path).ok().and_then(|text| parse(&text));
        Ok(Self { path, record })
    }

    pub fn candidate(&self) -> Option<&Candidate> {
        self.record.as_ref().map(|record| &record.candidate)
    }

    pub fn val_loss(&self) -> Option<f64> {
        self.record.as_ref().map(|record| record.val_loss)
    }

    pub fn screen_loss(&self) -> Option<f64> {
        self.record.as_ref().and_then(|record| record.screen_loss)
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

    pub fn promote_best(&mut self, trials: &[Trial], dry_run: bool) -> io::Result<bool> {
        let mut promoted = false;
        for trial in trials {
            promoted |= self.promote_trial(trial, dry_run)?;
        }
        Ok(promoted)
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

        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = fs::File::create(&self.path)?;
        writeln!(file, "VAL_LOSS={:.6}", record.val_loss)?;
        if let Some(steps) = record.completed_steps {
            writeln!(file, "COMPLETED_STEPS={steps}")?;
        }
        if let Some(elapsed_s) = record.elapsed_s {
            writeln!(file, "TRAIN_ELAPSED_S={elapsed_s:.6}")?;
        }
        if let Some(screen_loss) = record.screen_loss {
            writeln!(file, "SCREEN_LOSS={screen_loss:.6}")?;
        }
        if let Some(steps) = record.screen_completed_steps {
            writeln!(file, "SCREEN_COMPLETED_STEPS={steps}")?;
        }
        if let Some(elapsed_s) = record.screen_elapsed_s {
            writeln!(file, "SCREEN_ELAPSED_S={elapsed_s:.6}")?;
        }
        if let Some(reason) = &record.screen_reason {
            writeln!(file, "SCREEN_REASON={reason}")?;
        }
        writeln!(file, "LOG_PATH={}", record.log_path.display())?;
        writeln!(file, "GPT2_SEQ_LEN={DEFAULT_SEQ_LEN}")?;
        write_env(&mut file, record.candidate.build_env())?;
        write_env(&mut file, record.candidate.run_env())
    }
}

fn write_env(file: &mut fs::File, values: Vec<(&'static str, String)>) -> io::Result<()> {
    for (name, value) in values {
        writeln!(file, "{name}={value}")?;
    }
    Ok(())
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

fn parse(text: &str) -> Option<Record> {
    Some(Record {
        val_loss: value(text, "VAL_LOSS")?.parse().ok()?,
        completed_steps: value(text, "COMPLETED_STEPS").and_then(|value| value.parse().ok()),
        elapsed_s: value(text, "TRAIN_ELAPSED_S").and_then(|value| value.parse().ok()),
        screen_loss: value(text, "SCREEN_LOSS").and_then(|value| value.parse().ok()),
        screen_completed_steps: value(text, "SCREEN_COMPLETED_STEPS")
            .and_then(|value| value.parse().ok()),
        screen_elapsed_s: value(text, "SCREEN_ELAPSED_S").and_then(|value| value.parse().ok()),
        screen_reason: value(text, "SCREEN_REASON").map(ToString::to_string),
        log_path: PathBuf::from(value(text, "LOG_PATH").unwrap_or("")),
        candidate: Candidate {
            batch_size: value(text, "GPT2_BATCH_SIZE")?.parse().ok()?,
            n_layer: value(text, "GPT2_N_LAYER")?.parse().ok()?,
            n_embd: value(text, "GPT2_N_EMBD")?.parse().ok()?,
            n_head: value(text, "GPT2_N_HEAD")?.parse().ok()?,
            aurora_phases: value(text, "AURORA_MATRIX_PHASES")?.parse().ok()?,
            aurora_blocks: value(text, "AURORA_COOPERATIVE_BLOCKS")?.parse().ok()?,
            lr_scale: value(text, "TRAIN_LR_SCALE")?.parse().ok()?,
            adam_lr_scale: value(text, "TRAIN_ADAM_LR_SCALE")?.parse().ok()?,
            warmup_steps: value(text, "TRAIN_LR_WARMUP_STEPS")?.parse().ok()?,
            start_ratio: value(text, "TRAIN_LR_START_RATIO")?.parse().ok()?,
            amuse_beta1: value(text, "TRAIN_AMUSE_BETA1")?.parse().ok()?,
            amuse_rho: value(text, "TRAIN_AMUSE_RHO")?.parse().ok()?,
        },
    })
}

fn value<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    text.lines().find_map(|line| {
        let (key, value) = line.split_once('=')?;
        (key.trim() == name).then_some(value.trim())
    })
}
