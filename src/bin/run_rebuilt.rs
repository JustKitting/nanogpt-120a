use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let arch = std::env::var("CUDA_OXIDE_ARCH").unwrap_or_else(|_| "sm_120a".to_string());
    let build_status = Command::new("cargo")
        .args(["oxide", "build", "--arch", &arch])
        .status()
        .expect("failed to start cargo oxide build");

    if !build_status.success() {
        return ExitCode::from(build_status.code().unwrap_or(1) as u8);
    }

    let run_status = Command::new("./target/release/rust-kernels")
        .args(std::env::args().skip(1))
        .status()
        .expect("failed to start rebuilt rust-kernels binary");

    ExitCode::from(run_status.code().unwrap_or(1) as u8)
}
