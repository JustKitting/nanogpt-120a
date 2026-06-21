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
    pub trial_completed_steps: Option<usize>,
    pub trial_elapsed_s: Option<f64>,
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
            trial_completed_steps: trial.completed_steps,
            trial_elapsed_s: trial.elapsed_s,
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
        .filter_map(|obs| {
            obs.screen
                .and_then(|log| speed_row(&obs.candidate, log))
                .or_else(|| {
                    (obs.status == "rejected_screen")
                        .then_some(())
                        .and_then(|_| trial_speed_row(obs))
                })
        })
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
        .filter_map(|obs| {
            obs.full
                .and_then(|log| speed_row(&obs.candidate, log))
                .or_else(|| trial_speed_row(obs))
        })
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

fn trial_speed_row(obs: &Observation) -> Option<(Candidate, f64)> {
    let steps = obs.trial_completed_steps? as f64;
    let elapsed = obs.trial_elapsed_s?;
    (steps > 0.0 && elapsed > 0.0).then_some((
        obs.candidate.clone(),
        steps * obs.candidate.batch_size as f64 * SEQ_LEN / elapsed,
    ))
}

#[cfg(test)]
mod tests {
    use super::{Observation, full_speed_rows};
    use crate::sweep::candidate::Candidate;

    #[test]
    fn full_speed_uses_persisted_trial_elapsed_when_log_is_missing() {
        let rows = full_speed_rows(&[Observation {
            candidate: candidate(),
            status: "success".to_string(),
            screen: None,
            full: None,
            trial_val_loss: Some(4.0),
            trial_completed_steps: Some(100),
            trial_elapsed_s: Some(20.0),
        }]);

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1, 100.0 * 8.0 * 1024.0 / 20.0);
    }

    fn candidate() -> Candidate {
        Candidate {
            batch_size: 8,
            n_layer: 4,
            n_embd: 1024,
            n_head: 16,
            aurora_phases: 4,
            aurora_blocks: 80,
            lr_scale: 1.0,
            adam_lr_scale: 1.0,
            warmup_steps: 20,
            start_ratio: 0.1,
            amuse_beta1: 0.4,
            amuse_rho: 0.8,
        }
    }
}
