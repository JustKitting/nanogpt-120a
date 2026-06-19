use std::{env, fs, path::PathBuf};

fn main() {
    let baseline = Baseline::load();
    let cooperative_blocks = env_usize("AURORA_COOPERATIVE_BLOCKS")
        .or_else(|| baseline.usize("AURORA_COOPERATIVE_BLOCKS"))
        .unwrap_or(80);
    let matrix_phases = env_usize("AURORA_MATRIX_PHASES")
        .or_else(|| baseline.usize("AURORA_MATRIX_PHASES"))
        .unwrap_or(2);

    assert!(
        cooperative_blocks > 0,
        "AURORA_COOPERATIVE_BLOCKS must be > 0"
    );
    assert!(matrix_phases > 0, "AURORA_MATRIX_PHASES must be > 0");

    for name in ["AURORA_COOPERATIVE_BLOCKS", "AURORA_MATRIX_PHASES"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    println!(
        "cargo:rerun-if-changed={}",
        baseline_path().to_string_lossy()
    );

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

fn env_usize(name: &str) -> Option<usize> {
    env::var(name).ok().and_then(|value| value.parse().ok())
}

struct Baseline {
    text: String,
}

impl Baseline {
    fn load() -> Self {
        Self {
            text: fs::read_to_string(baseline_path()).unwrap_or_default(),
        }
    }

    fn usize(&self, name: &str) -> Option<usize> {
        self.value(name).and_then(|value| value.parse().ok())
    }

    fn value(&self, name: &str) -> Option<&str> {
        self.text.lines().find_map(|line| {
            let (key, value) = line.split_once('=')?;
            (key.trim() == name).then_some(value.trim())
        })
    }
}

fn baseline_path() -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"))
        .join("../../notes/sweep_baseline.env")
}
