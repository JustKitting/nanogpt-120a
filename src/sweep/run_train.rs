use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
    path::Path,
};

use super::{candidate::Candidate, config::SweepConfig, parse::RunResult, status};

mod command;
mod stage;

use command::training_command;
use stage::Stage;

pub fn run_screen_candidate(
    candidate: &Candidate,
    config: &SweepConfig,
    sweep_dir: &Path,
    trial_dir: &Path,
    trial_index: usize,
) -> std::io::Result<RunResult> {
    run_candidate_stage(
        candidate,
        config,
        sweep_dir,
        trial_dir,
        trial_index,
        Stage::Screen,
    )
}

pub fn run_candidate(
    candidate: &Candidate,
    config: &SweepConfig,
    sweep_dir: &Path,
    trial_dir: &Path,
    trial_index: usize,
) -> std::io::Result<RunResult> {
    run_candidate_stage(
        candidate,
        config,
        sweep_dir,
        trial_dir,
        trial_index,
        Stage::Full,
    )
}

fn run_candidate_stage(
    candidate: &Candidate,
    config: &SweepConfig,
    sweep_dir: &Path,
    trial_dir: &Path,
    trial_index: usize,
    stage: Stage,
) -> std::io::Result<RunResult> {
    let record = |event: &str, result: &RunResult| {
        status::record(sweep_dir, trial_dir, trial_index, candidate, event, result)
    };
    let record_stage = |suffix: &str, result: &RunResult| {
        record(&format!("{}_{suffix}", stage.event_prefix()), result)
    };
    let mut log = File::create(trial_dir.join(stage.log_name()))?;
    let mut command = training_command(candidate, config, stage);
    let mut child = command.spawn()?;
    let stdout = child.stdout.take().expect("stdout must be piped");
    let mut result = RunResult::default();
    record_stage("started", &result)?;
    for line in BufReader::new(stdout).lines() {
        let line = line?;
        let previous_steps = result.completed_steps;
        let previous_val_loss = result.val_loss;
        result.update(&line);
        println!("{line}");
        writeln!(log, "{line}")?;
        if result.completed_steps != previous_steps
            || result.val_loss != previous_val_loss
            || result.saw_nan
        {
            record_stage("progress", &result)?;
        }
        if result.saw_nan {
            writeln!(log, "sweep_early_stop=nan_detected")?;
            println!("sweep_early_stop=nan_detected");
            record("nan_detected", &result)?;
            let _ = child.kill();
            break;
        }
    }
    if let Some(stderr) = child.stderr.take() {
        for line in BufReader::new(stderr).lines() {
            writeln!(log, "{}", line?)?;
        }
    }
    child.wait()?;
    record_stage("exited", &result)?;
    Ok(result)
}
