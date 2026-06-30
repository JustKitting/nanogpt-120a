use std::{fs, path::Path};

use super::super::trial_record::{trial, trial_with_log};
use crate::sweep::{
    analysis,
    candidate::Candidate,
    config::SweepConfig,
    history::{self, Trial},
    parse::RunResult,
    run_build, run_train, screen_gate, status,
};

pub(in crate::sweep::runner) fn run_trial(
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
    let record = |event: &str, result: &RunResult| {
        status::record(sweep_dir, trial_dir, index, &candidate, event, result)
    };
    record("trial_started", &run_result)?;
    if config.dry_run {
        record("dry_run", &run_result)?;
        return Ok(empty_trial(candidate, "dry_run", trial_dir));
    }

    record("build_started", &run_result)?;
    let build_status =
        run_build::build_candidate(&candidate, config, &trial_dir.join("build.log"))?;
    if !build_status.success() {
        record("failed_build", &run_result)?;
        return Ok(empty_trial(candidate, "failed_build", trial_dir));
    }

    let screen_result =
        run_train::run_screen_candidate(&candidate, config, sweep_dir, trial_dir, index)?;
    let screen_decision = screen_gate::decide(&screen_result, screen_baseline, screen_score);
    screen_gate::write(&trial_dir.join("screen_decision.env"), &screen_decision)?;
    if !screen_decision.pass {
        record(
            &format!("rejected_screen_{}", screen_decision.reason),
            &screen_result,
        )?;
        return Ok(trial_with_log(
            candidate,
            "rejected_screen",
            RunResult::default(),
            screen_result,
            Some(screen_decision.reason),
            trial_dir,
            "screen.log",
        ));
    }
    record("screen_passed", &screen_result)?;

    run_result = run_train::run_candidate(&candidate, config, sweep_dir, trial_dir, index)?;
    let status_name = match (run_result.val_loss, run_result.saw_nan) {
        (Some(_), false) => "success",
        (Some(_), true) => "nan_with_val",
        (None, true) => "nan",
        (None, false) => "failed_run",
    };
    record(status_name, &run_result)?;
    Ok(trial(
        candidate,
        status_name,
        run_result,
        screen_result,
        Some(screen_decision.reason),
        trial_dir,
    ))
}

fn empty_trial(candidate: Candidate, status: &'static str, trial_dir: &Path) -> Trial {
    trial(
        candidate,
        status,
        RunResult::default(),
        RunResult::default(),
        None,
        trial_dir,
    )
}
