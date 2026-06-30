use super::super::{candidate::Candidate, history::Trial};
use super::log_files;

mod rows;

pub use rows::{full_quality_rows, screen_quality_rows, stability_rows};

#[derive(Clone)]
pub struct Observation {
    pub candidate: Candidate,
    pub status: String,
    pub screen: Option<LogMetrics>,
    pub full: Option<LogMetrics>,
    pub trial_val_loss: Option<f64>,
    pub trial_elapsed_s: Option<f64>,
    pub trial_screen_val_loss: Option<f64>,
    pub trial_screen_elapsed_s: Option<f64>,
    pub trial_screen_reason: Option<String>,
}

#[derive(Clone, Copy, Default)]
pub struct LogMetrics {
    pub val_loss: Option<f64>,
    pub completed_steps: Option<usize>,
    pub elapsed_s: Option<f64>,
    pub saw_nan: bool,
    pub panicked: bool,
}

pub fn observations(trials: &[Trial]) -> Vec<Observation> {
    trials
        .iter()
        .map(|trial| Observation {
            candidate: trial.candidate.clone(),
            status: trial.status.clone(),
            screen: log_files::read_log(log_files::screen_path(trial)),
            full: log_files::read_log(log_files::full_path(trial)),
            trial_val_loss: trial.val_loss,
            trial_elapsed_s: trial.elapsed_s,
            trial_screen_val_loss: trial.screen_val_loss,
            trial_screen_elapsed_s: trial.screen_elapsed_s,
            trial_screen_reason: trial.screen_reason.clone(),
        })
        .collect()
}
