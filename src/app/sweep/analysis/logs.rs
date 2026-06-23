use super::super::{candidate::Candidate, history::Trial};
use super::log_files;

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

pub fn screen_quality_rows(
    observations: &[Observation],
    screen_max_seconds: f64,
) -> Vec<(Candidate, f64)> {
    observations
        .iter()
        .filter_map(|obs| {
            obs.screen
                .and_then(|screen| {
                    screen_time_matches(screen.elapsed_s, screen_max_seconds)
                        .then_some((obs.candidate.clone(), -screen.val_loss?))
                })
                .or_else(|| {
                    screen_time_matches(obs.trial_screen_elapsed_s, screen_max_seconds)
                        .then_some((obs.candidate.clone(), -obs.trial_screen_val_loss?))
                })
        })
        .collect()
}

pub fn full_quality_rows(observations: &[Observation], max_seconds: f64) -> Vec<(Candidate, f64)> {
    observations
        .iter()
        .filter_map(|obs| {
            if obs.status != "success" {
                return None;
            }
            obs.full
                .and_then(|full| {
                    screen_time_matches(full.elapsed_s, max_seconds)
                        .then_some((obs.candidate.clone(), -full.val_loss?))
                })
                .or_else(|| {
                    screen_time_matches(obs.trial_elapsed_s, max_seconds)
                        .then_some((obs.candidate.clone(), -obs.trial_val_loss?))
                })
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
                || screen_reason_failed(obs.trial_screen_reason.as_deref())
                || obs.screen.is_some_and(|log| log.saw_nan || log.panicked)
                || obs.full.is_some_and(|log| log.saw_nan || log.panicked);
            (obs.candidate.clone(), if failed { 0.0 } else { 1.0 })
        })
        .collect()
}

fn screen_reason_failed(reason: Option<&str>) -> bool {
    matches!(reason, Some("nan" | "missing_val_loss"))
}

fn screen_time_matches(elapsed_s: Option<f64>, screen_max_seconds: f64) -> bool {
    let Some(elapsed_s) = elapsed_s else {
        return false;
    };
    elapsed_s.is_finite()
        && elapsed_s >= screen_max_seconds * 0.8
        && elapsed_s <= screen_max_seconds * 1.25
}

#[cfg(test)]
mod tests {
    use super::Observation;
    use crate::sweep::candidate::Candidate;

    #[test]
    fn screen_quality_uses_persisted_screen_loss_when_log_is_missing() {
        let rows = super::screen_quality_rows(
            &[Observation {
                candidate: candidate(),
                status: "rejected_screen".to_string(),
                screen: None,
                full: None,
                trial_val_loss: None,
                trial_elapsed_s: None,
                trial_screen_val_loss: Some(5.25),
                trial_screen_elapsed_s: Some(90.0),
                trial_screen_reason: Some("screen_loss_worse".to_string()),
            }],
            90.0,
        );

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1, -5.25);
    }

    #[test]
    fn screen_quality_ignores_different_time_budget() {
        let rows = super::screen_quality_rows(
            &[Observation {
                candidate: candidate(),
                status: "rejected_screen".to_string(),
                screen: None,
                full: None,
                trial_val_loss: None,
                trial_elapsed_s: None,
                trial_screen_val_loss: Some(3.25),
                trial_screen_elapsed_s: Some(180.0),
                trial_screen_reason: Some("screen_loss_improved".to_string()),
            }],
            30.0,
        );

        assert!(rows.is_empty());
    }

    #[test]
    fn stability_marks_missing_val_loss_screen_rejection_as_failure() {
        let rows = super::stability_rows(&[Observation {
            candidate: candidate(),
            status: "rejected_screen".to_string(),
            screen: None,
            full: None,
            trial_val_loss: None,
            trial_elapsed_s: None,
            trial_screen_val_loss: None,
            trial_screen_elapsed_s: Some(180.0),
            trial_screen_reason: Some("missing_val_loss".to_string()),
        }]);

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1, 0.0);
    }

    #[test]
    fn stability_keeps_worse_screen_loss_as_survived() {
        let rows = super::stability_rows(&[Observation {
            candidate: candidate(),
            status: "rejected_screen".to_string(),
            screen: None,
            full: None,
            trial_val_loss: None,
            trial_elapsed_s: None,
            trial_screen_val_loss: Some(5.25),
            trial_screen_elapsed_s: Some(90.0),
            trial_screen_reason: Some("screen_loss_worse".to_string()),
        }]);

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1, 1.0);
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
            nextlat_lr_scale: 1.0,
            warmup_steps: 20,
            start_ratio: 0.1,
            amuse_beta1: 0.4,
            amuse_rho: 0.8,
        }
    }
}
