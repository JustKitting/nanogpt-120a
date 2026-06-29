use std::collections::HashSet;

use super::{
    history::{History, Trial},
    rng::SweepRng,
};

pub(super) fn sync_shared_history(
    shared_history: &mut History,
    local_trials: &[Trial],
    dry_run: bool,
) -> std::io::Result<()> {
    if dry_run {
        return Ok(());
    }
    for trial in local_trials
        .iter()
        .filter(|trial| trial.status != "dry_run")
    {
        shared_history.append_unique(trial.clone())?;
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn all_trials(shared_trials: &[Trial], local_trials: &[Trial]) -> Vec<Trial> {
    all_trials_with_baseline(None, shared_trials, local_trials)
}

pub(super) fn all_trials_with_baseline(
    baseline: Option<&Trial>,
    shared_trials: &[Trial],
    local_trials: &[Trial],
) -> Vec<Trial> {
    let mut seen = HashSet::new();
    let mut trials = Vec::new();
    for trial in baseline
        .into_iter()
        .chain(shared_trials.iter())
        .chain(local_trials)
    {
        if seen.insert(trial.candidate.key()) {
            trials.push(trial.clone());
        }
    }
    trials
}

pub(super) fn seen_keys(trials: &[Trial]) -> HashSet<String> {
    trials.iter().map(|trial| trial.candidate.key()).collect()
}

pub(super) fn sweep_rng(seed: u64, completed_trials: usize) -> SweepRng {
    SweepRng::new(seed ^ completed_trials as u64)
}
