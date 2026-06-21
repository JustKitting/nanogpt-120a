use super::super::{candidate::Candidate, history::Trial};
use super::log_files;

const SEQ_LEN: f64 = 1024.0;

#[derive(Clone)]
pub struct Observation {
    pub candidate: Candidate,
    pub status: String,
    pub screen: Option<LogMetrics>,
    pub full: Option<LogMetrics>,
    pub trial_val_loss: Option<f64>,
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
        })
        .collect()
}

pub fn screen_quality_rows(
    observations: &[Observation],
    screen_steps: usize,
) -> Vec<(Candidate, f64)> {
    observations
        .iter()
        .filter_map(|obs| {
            let screen = obs.screen?;
            (screen.completed_steps.unwrap_or(0) >= screen_steps)
                .then_some((obs.candidate.clone(), -screen.val_loss?))
        })
        .collect()
}

pub fn screen_speed_rows(observations: &[Observation]) -> Vec<(Candidate, f64)> {
    observations
        .iter()
        .filter_map(|obs| speed_row(&obs.candidate, obs.screen?))
        .collect()
}

pub fn full_quality_rows(observations: &[Observation]) -> Vec<(Candidate, f64)> {
    observations
        .iter()
        .filter_map(|obs| {
            let val_loss = obs
                .trial_val_loss
                .or(obs.full.and_then(|full| full.val_loss))?;
            (obs.status == "success").then_some((obs.candidate.clone(), -val_loss))
        })
        .collect()
}

pub fn full_speed_rows(observations: &[Observation]) -> Vec<(Candidate, f64)> {
    observations
        .iter()
        .filter(|obs| obs.status == "success")
        .filter_map(|obs| speed_row(&obs.candidate, obs.full?))
        .collect()
}

pub fn stability_rows(observations: &[Observation]) -> Vec<(Candidate, f64)> {
    observations
        .iter()
        .filter(|obs| obs.status != "dry_run")
        .map(|obs| {
            let failed = obs.status.starts_with("nan")
                || obs.status == "failed_build"
                || obs.status == "failed_run"
                || obs.screen.is_some_and(|log| log.saw_nan || log.panicked)
                || obs.full.is_some_and(|log| log.saw_nan || log.panicked);
            (obs.candidate.clone(), if failed { 0.0 } else { 1.0 })
        })
        .collect()
}

fn speed_row(candidate: &Candidate, log: LogMetrics) -> Option<(Candidate, f64)> {
    let steps = log.completed_steps? as f64;
    let elapsed = log.elapsed_s?;
    (steps > 0.0 && elapsed > 0.0).then_some((
        candidate.clone(),
        steps * candidate.batch_size as f64 * SEQ_LEN / elapsed,
    ))
}
