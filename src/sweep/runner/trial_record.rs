use std::fs;
use std::path::{Path, PathBuf};

use crate::sweep::candidate::Candidate;
use crate::sweep::env_file::parsed;
use crate::sweep::history::Trial;
use crate::sweep::parse::RunResult;

#[cfg(test)]
mod tests;

pub(super) fn current_baseline_trial(
    screen_trial: Option<&Trial>,
    measured_trial: Option<Trial>,
) -> Option<Trial> {
    screen_trial.cloned().or(measured_trial)
}

pub(super) fn trial(
    candidate: Candidate,
    status: &str,
    train: RunResult,
    screen: RunResult,
    screen_reason: Option<&str>,
    trial_dir: &Path,
) -> Trial {
    trial_with_log(
        candidate,
        status,
        train,
        screen,
        screen_reason,
        trial_dir,
        "train.log",
    )
}

pub(super) fn trial_with_log(
    candidate: Candidate,
    status: &str,
    train: RunResult,
    screen: RunResult,
    screen_reason: Option<&str>,
    trial_dir: &Path,
    log_name: &str,
) -> Trial {
    Trial {
        candidate,
        status: status.to_string(),
        val_loss: train.val_loss,
        completed_steps: train.completed_steps,
        log_path: PathBuf::from(trial_dir).join(log_name),
        elapsed_s: train.last_elapsed_s,
        screen_val_loss: screen.val_loss,
        screen_completed_steps: screen.completed_steps,
        screen_elapsed_s: screen.last_elapsed_s,
        screen_reason: screen_reason.map(ToString::to_string),
    }
}

pub(super) fn promoted_screen_loss(trial: &Trial) -> Option<f64> {
    let text = fs::read_to_string(trial.log_path.with_file_name("screen_decision.env")).ok()?;
    parsed(&text, "SCREEN_LOSS")
}
