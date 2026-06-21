use std::{
    fs::File,
    io::Write,
    path::Path,
    process::{Command, ExitStatus},
};

use super::{candidate::Candidate, config::SweepConfig};

pub fn build_candidate(
    candidate: &Candidate,
    config: &SweepConfig,
    log_path: &Path,
) -> std::io::Result<ExitStatus> {
    let mut log = File::create(log_path)?;

    let mut command = cargo_with_candidate_env(candidate);
    command.args(["oxide", "build", "--arch", &config.arch]);
    let output = command.output()?;
    log.write_all(&output.stdout)?;
    log.write_all(&output.stderr)?;
    if !output.status.success() {
        return Ok(output.status);
    }

    let mut command = cargo_with_candidate_env(candidate);
    command.args(["build", "--release", "--bin", "rust-kernels"]);
    let output = command.output()?;
    log.write_all(&output.stdout)?;
    log.write_all(&output.stderr)?;
    Ok(output.status)
}

fn cargo_with_candidate_env(candidate: &Candidate) -> Command {
    let mut command = Command::new("cargo");
    for (name, value) in candidate.build_env() {
        command.env(name, value);
    }
    command
}
