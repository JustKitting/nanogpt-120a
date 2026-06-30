use crate::sweep::candidate::Candidate;

use super::Observation;

#[cfg(test)]
mod tests;

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
