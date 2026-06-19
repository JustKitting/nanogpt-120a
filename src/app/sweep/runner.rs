use std::{
    fs,
    path::{Path, PathBuf},
};

use time::OffsetDateTime;

use super::{
    candidate::Candidate,
    chain,
    config::SweepConfig,
    history::{self, History, Trial},
    optimizer, run_build, run_train,
};

pub fn run(config: SweepConfig) -> Result<(), Box<dyn std::error::Error>> {
    let sweep_dir = config.sweep_dir.clone().unwrap_or_else(default_sweep_dir);
    fs::create_dir_all(&sweep_dir)?;
    let mut history = History::load(sweep_dir.join("trials.tsv"))?;
    let mut shared_history = History::load(config.seed_history.clone())?;
    chain::sync_shared_history(&mut shared_history, &history.trials, config.dry_run)?;
    let mut rng = chain::sweep_rng(config.seed, history.trials.len());

    for index in history.trials.len()..config.trials {
        let all_trials = chain::all_trials(&shared_history.trials, &history.trials);
        let seen = chain::seen_keys(&all_trials);
        let candidate = optimizer::propose(
            &all_trials,
            &seen,
            &mut rng,
            config.random_trials,
            config.candidate_samples,
        );
        let trial_dir = sweep_dir.join(format!("trial_{index:04}"));
        let trial = run_trial(&trial_dir, candidate, &config)?;
        history.append_unique(trial.clone())?;
        if !config.dry_run {
            shared_history.append_unique(trial)?;
        }
    }
    Ok(())
}

fn run_trial(
    trial_dir: &Path,
    candidate: Candidate,
    config: &SweepConfig,
) -> Result<Trial, Box<dyn std::error::Error>> {
    fs::create_dir_all(trial_dir)?;
    history::write_candidate(&trial_dir.join("candidate.env"), &candidate)?;
    if config.dry_run {
        return Ok(trial(candidate, "dry_run", None, None, trial_dir));
    }

    let build_status =
        run_build::build_candidate(&candidate, config, &trial_dir.join("build.log"))?;
    if !build_status.success() {
        return Ok(trial(candidate, "failed_build", None, None, trial_dir));
    }

    let run_result = run_train::run_candidate(&candidate, config, trial_dir)?;
    let status = match (run_result.val_loss, run_result.saw_nan) {
        (Some(_), false) => "success",
        (Some(_), true) => "nan_with_val",
        (None, true) => "nan",
        (None, false) => "failed_run",
    };
    Ok(trial(
        candidate,
        status,
        run_result.val_loss,
        run_result.completed_steps,
        trial_dir,
    ))
}

fn trial(
    candidate: Candidate,
    status: &str,
    val_loss: Option<f64>,
    completed_steps: Option<usize>,
    trial_dir: &Path,
) -> Trial {
    Trial {
        candidate,
        status: status.to_string(),
        val_loss,
        completed_steps,
        log_path: PathBuf::from(trial_dir).join("train.log"),
    }
}

fn default_sweep_dir() -> PathBuf {
    PathBuf::from("target/sweeps").join(utc_stamp())
}

fn utc_stamp() -> String {
    let now = OffsetDateTime::now_utc();
    format!(
        "{:04}{:02}{:02}_{:02}{:02}{:02}Z",
        now.year(),
        now.month() as u8,
        now.day(),
        now.hour(),
        now.minute(),
        now.second()
    )
}
