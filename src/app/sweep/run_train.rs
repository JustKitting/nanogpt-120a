use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Command, Stdio},
};

use super::{candidate::Candidate, config::SweepConfig, parse::RunResult, status};

pub fn run_candidate(
    candidate: &Candidate,
    config: &SweepConfig,
    sweep_dir: &Path,
    trial_dir: &Path,
    trial_index: usize,
) -> std::io::Result<RunResult> {
    let mut log = File::create(trial_dir.join("train.log"))?;
    let mut command = Command::new("./target/release/rust-kernels");
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    command.env("TRAIN_DATASET", &config.dataset);
    command.env("TRAIN_MAX_SECONDS", format!("{:.3}", config.max_seconds));
    command.env("TRAIN_LOG_INTERVAL", config.log_interval.to_string());
    command.env("TRAIN_RUN_DIR", trial_dir.join("run"));
    if let Some(device) = &config.cuda_device {
        command.env("CUDA_DEVICE_INDEX", device);
    }
    for (name, value) in candidate.run_env() {
        command.env(name, value);
    }

    let mut child = command.spawn()?;
    let stdout = child.stdout.take().expect("stdout must be piped");
    let mut result = RunResult::default();
    status::record(
        sweep_dir,
        trial_dir,
        trial_index,
        candidate,
        "training_started",
        &result,
    )?;
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
            status::record(
                sweep_dir,
                trial_dir,
                trial_index,
                candidate,
                "training_progress",
                &result,
            )?;
        }
        if result.saw_nan {
            writeln!(log, "sweep_early_stop=nan_detected")?;
            println!("sweep_early_stop=nan_detected");
            status::record(
                sweep_dir,
                trial_dir,
                trial_index,
                candidate,
                "nan_detected",
                &result,
            )?;
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
    status::record(
        sweep_dir,
        trial_dir,
        trial_index,
        candidate,
        "training_exited",
        &result,
    )?;
    Ok(result)
}
