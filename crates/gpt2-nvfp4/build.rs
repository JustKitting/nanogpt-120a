use std::{env, fs, path::PathBuf};

fn main() {
    let baseline = Baseline::load();
    let seq_len = env_usize("GPT2_SEQ_LEN")
        .or_else(|| baseline.usize("GPT2_SEQ_LEN"))
        .unwrap_or(4096);
    let batch_size = env_usize("GPT2_BATCH_SIZE")
        .or_else(|| baseline.usize("GPT2_BATCH_SIZE"))
        .unwrap_or(4);
    let n_layer = env_usize("GPT2_N_LAYER")
        .or_else(|| baseline.usize("GPT2_N_LAYER"))
        .unwrap_or(8);
    let n_head = env_usize("GPT2_N_HEAD")
        .or_else(|| baseline.usize("GPT2_N_HEAD"))
        .unwrap_or(32);
    let n_embd = env_usize("GPT2_N_EMBD")
        .or_else(|| baseline.usize("GPT2_N_EMBD"))
        .unwrap_or(2048);

    assert!(seq_len > 0, "GPT2_SEQ_LEN must be > 0");
    assert!(batch_size > 0, "GPT2_BATCH_SIZE must be > 0");
    assert!(n_layer >= 4, "GPT2_N_LAYER must be >= 4");
    assert!(n_head > 0, "GPT2_N_HEAD must be > 0");
    assert_eq!(
        n_embd % n_head,
        0,
        "GPT2_N_EMBD must be divisible by GPT2_N_HEAD"
    );

    for name in [
        "GPT2_SEQ_LEN",
        "GPT2_BATCH_SIZE",
        "GPT2_N_LAYER",
        "GPT2_N_HEAD",
        "GPT2_N_EMBD",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    println!(
        "cargo:rerun-if-changed={}",
        baseline_path().to_string_lossy()
    );

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    fs::write(
        out.join("gpt2_shape.rs"),
        format!(
            "pub const GPT2_SEQ_LEN: usize = {seq_len};\n\
             pub const GPT2_BATCH_SIZE: usize = {batch_size};\n\
             pub const GPT2_N_LAYER: usize = {n_layer};\n\
             pub const GPT2_N_HEAD: usize = {n_head};\n\
             pub const GPT2_N_EMBD: usize = {n_embd};\n"
        ),
    )
    .expect("failed to write generated GPT-2 shape");
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
