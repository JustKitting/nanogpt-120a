use std::{env, fs, path::PathBuf};

fn main() {
    let seq_len = env_usize("GPT2_SEQ_LEN", 1024);
    let batch_size = env_usize("GPT2_BATCH_SIZE", 4);
    let n_layer = env_usize("GPT2_N_LAYER", 2);
    let n_head = env_usize("GPT2_N_HEAD", 12);
    let n_embd = env_usize("GPT2_N_EMBD", 1536);

    assert!(seq_len > 0, "GPT2_SEQ_LEN must be > 0");
    assert!(batch_size > 0, "GPT2_BATCH_SIZE must be > 0");
    assert!(n_layer > 0, "GPT2_N_LAYER must be > 0");
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

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
