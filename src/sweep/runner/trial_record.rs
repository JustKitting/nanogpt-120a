use std::fs;
use std::path::{Path, PathBuf};

use crate::sweep::candidate::Candidate;
use crate::sweep::history::Trial;

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
    val_loss: Option<f64>,
    completed_steps: Option<usize>,
    elapsed_s: Option<f64>,
    screen_val_loss: Option<f64>,
    screen_completed_steps: Option<usize>,
    screen_elapsed_s: Option<f64>,
    screen_reason: Option<&str>,
    trial_dir: &Path,
) -> Trial {
    trial_with_log(
        candidate,
        status,
        val_loss,
        completed_steps,
        elapsed_s,
        screen_val_loss,
        screen_completed_steps,
        screen_elapsed_s,
        screen_reason,
        trial_dir,
        "train.log",
    )
}

pub(super) fn trial_with_log(
    candidate: Candidate,
    status: &str,
    val_loss: Option<f64>,
    completed_steps: Option<usize>,
    elapsed_s: Option<f64>,
    screen_val_loss: Option<f64>,
    screen_completed_steps: Option<usize>,
    screen_elapsed_s: Option<f64>,
    screen_reason: Option<&str>,
    trial_dir: &Path,
    log_name: &str,
) -> Trial {
    Trial {
        candidate,
        status: status.to_string(),
        val_loss,
        completed_steps,
        log_path: PathBuf::from(trial_dir).join(log_name),
        elapsed_s,
        screen_val_loss,
        screen_completed_steps,
        screen_elapsed_s,
        screen_reason: screen_reason.map(ToString::to_string),
    }
}

pub(super) fn promoted_screen_loss(trial: &Trial) -> Option<f64> {
    let text = fs::read_to_string(trial.log_path.with_file_name("screen_decision.env")).ok()?;
    value(&text, "SCREEN_LOSS")?.parse().ok()
}

fn value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines().find_map(|line| {
        let (name, value) = line.split_once('=')?;
        (name == key).then_some(value)
    })
}
