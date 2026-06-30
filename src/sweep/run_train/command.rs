use std::process::{Command, Stdio};

use super::stage::Stage;
use crate::sweep::{candidate::Candidate, config::SweepConfig};

pub(super) fn training_command(
    candidate: &Candidate,
    config: &SweepConfig,
    stage: Stage,
) -> Command {
    let mut command = Command::new("./target/release/rust-kernels");
    command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("TRAIN_DATASET", &config.dataset)
        .env(
            "TRAIN_MAX_SECONDS",
            format!("{:.3}", stage.max_seconds(config)),
        )
        .env("TRAIN_LOG_INTERVAL", config.log_interval.to_string());
    if let Some(device) = &config.cuda_device {
        command.env("CUDA_DEVICE_INDEX", device);
    }
    for (name, value) in candidate.run_env() {
        command.env(name, value);
    }
    command
}
