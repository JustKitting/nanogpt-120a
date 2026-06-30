use std::fs;
use std::path::Path;

use super::trial_record::{trial, trial_with_log};
use crate::sweep::{
    analysis,
    baseline::Baseline,
    candidate::Candidate,
    config::SweepConfig,
    history::{self, Trial},
    parse::RunResult,
    run_build, run_train, screen_gate, status,
};

pub(super) fn screen_baseline(
    baseline: &Baseline,
    config: &SweepConfig,
    sweep_dir: &Path,
) -> Result<Option<Trial>, Box<dyn std::error::Error>> {
    let Some(mut trial) = baseline.measured_trial() else {
        return Ok(None);
    };
    let candidate = trial.candidate.clone();
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
    if let Some(val_loss) = result.val_loss {
        println!(
            "sweep_screen_baseline val_loss={val_loss:.6} completed_steps={}",
            result
                .completed_steps
                .map(|value| value.to_string())
                .unwrap_or_default()
        );
        trial.screen_val_loss = Some(val_loss);
        trial.screen_completed_steps = result.completed_steps;
        trial.screen_elapsed_s = result.last_elapsed_s;
        trial.screen_reason = Some("screen_baseline".to_string());
        trial.log_path = trial_dir.join("screen.log");
        Ok(Some(trial))
    } else {
        println!("sweep_screen_baseline_failed=run");
        Ok(None)
    }
}

pub(super) fn run_trial(
    index: usize,
    sweep_dir: &Path,
    trial_dir: &Path,
    candidate: Candidate,
    config: &SweepConfig,
    screen_baseline: Option<f64>,
    screen_score: Option<&analysis::CandidateScore>,
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
        return Ok(trial(
            candidate, "dry_run", None, None, None, None, None, None, None, trial_dir,
        ));
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
        return Ok(trial(
            candidate,
            "failed_build",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            trial_dir,
        ));
    }

    let screen_result =
        run_train::run_screen_candidate(&candidate, config, sweep_dir, trial_dir, index)?;
    let screen_decision = screen_gate::decide(&screen_result, screen_baseline, screen_score);
    screen_gate::write(&trial_dir.join("screen_decision.env"), &screen_decision)?;
    if !screen_decision.pass {
        status::record(
            sweep_dir,
            trial_dir,
            index,
            &candidate,
            &format!("rejected_screen_{}", screen_decision.reason),
            &screen_result,
        )?;
        return Ok(trial_with_log(
            candidate,
            "rejected_screen",
            None,
            screen_result.completed_steps,
            screen_result.last_elapsed_s,
            screen_result.val_loss,
            screen_result.completed_steps,
            screen_result.last_elapsed_s,
            Some(screen_decision.reason),
            trial_dir,
            "screen.log",
        ));
    }
    status::record(
        sweep_dir,
        trial_dir,
        index,
        &candidate,
        "screen_passed",
        &screen_result,
    )?;

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
        run_result.last_elapsed_s,
        screen_result.val_loss,
        screen_result.completed_steps,
        screen_result.last_elapsed_s,
        Some(screen_decision.reason),
        trial_dir,
    ))
}
