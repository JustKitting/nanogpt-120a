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
    let mut command = Command::new("cargo");
    command.args(["oxide", "build", "--arch", &config.arch]);
    for (name, value) in candidate.build_env() {
        command.env(name, value);
    }
    let output = command.output()?;
    let mut log = File::create(log_path)?;
    log.write_all(&output.stdout)?;
    log.write_all(&output.stderr)?;
    Ok(output.status)
}
