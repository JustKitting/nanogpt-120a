use std::{env, fs, path::PathBuf};

pub fn env_usize(name: &str) -> Option<usize> {
    env::var(name).ok().and_then(|value| value.parse().ok())
}

pub fn emit_rerun_metadata(env_names: &[&str]) {
    for name in env_names {
        println!("cargo:rerun-if-env-changed={name}");
    }
    println!(
        "cargo:rerun-if-changed={}",
        baseline_path().to_string_lossy()
    );
}

pub struct Baseline {
    text: String,
}

impl Baseline {
    pub fn load() -> Self {
        Self {
            text: fs::read_to_string(baseline_path()).unwrap_or_default(),
        }
    }

    pub fn usize(&self, name: &str) -> Option<usize> {
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
