use std::{
    fs,
    path::{Path, PathBuf},
};

use time::OffsetDateTime;

use super::{
    baseline::Baseline,
    candidate::Candidate,
    chain,
    config::SweepConfig,
    history::{self, History, Trial},
    optimizer,
    parse::RunResult,
    run_build, run_train, status,
};

pub fn run(config: SweepConfig) -> Result<(), Box<dyn std::error::Error>> {
    let sweep_dir = config.sweep_dir.clone().unwrap_or_else(default_sweep_dir);
    fs::create_dir_all(&sweep_dir)?;
    let mut history = History::load(sweep_dir.join("trials.tsv"))?;
    let mut shared_history = History::load(config.seed_history.clone())?;
    chain::sync_shared_history(&mut shared_history, &history.trials, config.dry_run)?;
    let mut baseline = Baseline::load(config.baseline.clone())?;
    let initial_trials = chain::all_trials(&shared_history.trials, &history.trials);
    if baseline.promote_best(&initial_trials, config.dry_run)? {
        println!(
            "sweep_baseline_promoted val_loss={:.6} key={} path={}",
            baseline.val_loss().unwrap_or(f64::NAN),
            baseline
                .candidate()
                .map(|candidate| candidate.key())
                .unwrap_or_default(),
            config.baseline.display()
        );
    }
    let screen_baseline = screen_baseline(&baseline, &config, &sweep_dir)?;
    let mut rng = chain::sweep_rng(config.seed, history.trials.len());

    for index in history.trials.len()..config.trials {
        let baseline_trial = baseline.measured_trial();
        let all_trials = chain::all_trials_with_baseline(
            baseline_trial.as_ref(),
            &shared_history.trials,
            &history.trials,
        );
        let seen = chain::seen_keys(&all_trials);
        let candidate = optimizer::propose(
            &all_trials,
            &seen,
            &mut rng,
            config.random_trials,
            config.candidate_samples,
            baseline.candidate(),
        );
        let trial_dir = sweep_dir.join(format!("trial_{index:04}"));
        println!("sweep_trial_begin index={index} key={}", candidate.key());
        let trial = run_trial(
            index,
            &sweep_dir,
            &trial_dir,
            candidate,
            &config,
            screen_baseline,
        )?;
        println!(
            "sweep_trial_end index={index} status={} val_loss={} completed_steps={} log_path={}",
            trial.status,
            trial
                .val_loss
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "NaN".to_string()),
            trial
                .completed_steps
                .map(|value| value.to_string())
                .unwrap_or_default(),
            trial.log_path.display()
        );
        history.append_unique(trial.clone())?;
        let promoted = baseline.promote_trial(&trial, config.dry_run)?;
        if !config.dry_run {
            shared_history.append_unique(trial)?;
        }
        if promoted {
            println!(
                "sweep_baseline_promoted val_loss={:.6} key={} path={}",
                baseline.val_loss().unwrap_or(f64::NAN),
                baseline
                    .candidate()
                    .map(|candidate| candidate.key())
                    .unwrap_or_default(),
                config.baseline.display()
            );
        }
    }
    Ok(())
}

fn screen_baseline(
    baseline: &Baseline,
    config: &SweepConfig,
    sweep_dir: &Path,
) -> Result<Option<f64>, Box<dyn std::error::Error>> {
    let Some(candidate) = baseline.candidate().cloned() else {
        return Ok(None);
    };
    if config.dry_run {
        return Ok(None);
    }

    let trial_dir = sweep_dir.join("screen_baseline");
    fs::create_dir_all(&trial_dir)?;
    let build_status =
        run_build::build_candidate(&candidate, config, &trial_dir.join("build.log"))?;
    if !build_status.success() {
        println!("sweep_screen_baseline_failed=build");
        return Ok(None);
    }

    let result = run_train::run_screen_candidate(&candidate, config, sweep_dir, &trial_dir, 0)?;
    if result.completed_steps.unwrap_or(0) < config.screen_steps {
        println!("sweep_screen_baseline_failed=incomplete");
        return Ok(None);
    }

    if let Some(val_loss) = result.val_loss {
        println!(
            "sweep_screen_baseline val_loss={val_loss:.6} completed_steps={}",
            result
                .completed_steps
                .map(|value| value.to_string())
                .unwrap_or_default()
        );
        Ok(Some(val_loss))
    } else {
        println!("sweep_screen_baseline_failed=run");
        Ok(None)
    }
}

fn run_trial(
    index: usize,
    sweep_dir: &Path,
    trial_dir: &Path,
    candidate: Candidate,
    config: &SweepConfig,
    screen_baseline: Option<f64>,
) -> Result<Trial, Box<dyn std::error::Error>> {
    fs::create_dir_all(trial_dir)?;
    history::write_candidate(&trial_dir.join("candidate.env"), &candidate)?;
    let mut run_result = RunResult::default();
    status::record(
        sweep_dir,
        trial_dir,
        index,
        &candidate,
        "trial_started",
        &run_result,
    )?;
    if config.dry_run {
        status::record(
            sweep_dir,
            trial_dir,
            index,
            &candidate,
            "dry_run",
            &run_result,
        )?;
        return Ok(trial(candidate, "dry_run", None, None, trial_dir));
    }

    status::record(
        sweep_dir,
        trial_dir,
        index,
        &candidate,
        "build_started",
        &run_result,
    )?;
    let build_status =
        run_build::build_candidate(&candidate, config, &trial_dir.join("build.log"))?;
    if !build_status.success() {
        status::record(
            sweep_dir,
            trial_dir,
            index,
            &candidate,
            "failed_build",
            &run_result,
        )?;
        return Ok(trial(candidate, "failed_build", None, None, trial_dir));
    }

    let screen_result =
        run_train::run_screen_candidate(&candidate, config, sweep_dir, trial_dir, index)?;
    if !passes_screen(&screen_result, screen_baseline, config.screen_steps) {
        status::record(
            sweep_dir,
            trial_dir,
            index,
            &candidate,
            "rejected_screen",
            &screen_result,
        )?;
        return Ok(trial_with_log(
            candidate,
            "rejected_screen",
            None,
            screen_result.completed_steps,
            trial_dir,
            "screen.log",
        ));
    }

    run_result = run_train::run_candidate(&candidate, config, sweep_dir, trial_dir, index)?;
    let status_name = match (run_result.val_loss, run_result.saw_nan) {
        (Some(_), false) => "success",
        (Some(_), true) => "nan_with_val",
        (None, true) => "nan",
        (None, false) => "failed_run",
    };
    status::record(
        sweep_dir,
        trial_dir,
        index,
        &candidate,
        status_name,
        &run_result,
    )?;
    Ok(trial(
        candidate,
        status_name,
        run_result.val_loss,
        run_result.completed_steps,
        trial_dir,
    ))
}

fn passes_screen(result: &RunResult, baseline_loss: Option<f64>, screen_steps: usize) -> bool {
    if result.completed_steps.unwrap_or(0) < screen_steps {
        return false;
    }
    let Some(screen_loss) = result.val_loss else {
        return false;
    };
    baseline_loss
        .map(|baseline_loss| screen_loss < baseline_loss)
        .unwrap_or(true)
}

fn trial(
    candidate: Candidate,
    status: &str,
    val_loss: Option<f64>,
    completed_steps: Option<usize>,
    trial_dir: &Path,
) -> Trial {
    trial_with_log(
        candidate,
        status,
        val_loss,
        completed_steps,
        trial_dir,
        "train.log",
    )
}

fn trial_with_log(
    candidate: Candidate,
    status: &str,
    val_loss: Option<f64>,
    completed_steps: Option<usize>,
    trial_dir: &Path,
    log_name: &str,
) -> Trial {
    Trial {
        candidate,
        status: status.to_string(),
        val_loss,
        completed_steps,
        log_path: PathBuf::from(trial_dir).join(log_name),
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
