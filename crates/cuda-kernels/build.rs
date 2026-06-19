use std::{env, fs, path::PathBuf};

fn main() {
    let cooperative_blocks = env_usize("AURORA_COOPERATIVE_BLOCKS", 180);
    let matrix_phases = env_usize("AURORA_MATRIX_PHASES", 8);

    assert!(
        cooperative_blocks > 0,
        "AURORA_COOPERATIVE_BLOCKS must be > 0"
    );
    assert!(matrix_phases > 0, "AURORA_MATRIX_PHASES must be > 0");

    for name in ["AURORA_COOPERATIVE_BLOCKS", "AURORA_MATRIX_PHASES"] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    fs::write(
        out.join("optimizer_config.rs"),
        format!(
            "pub const AURORA_COOPERATIVE_BLOCKS: usize = {cooperative_blocks};\n\
             pub const AURORA_MATRIX_PHASES: usize = {matrix_phases};\n"
        ),
    )
    .expect("failed to write generated optimizer config");
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
