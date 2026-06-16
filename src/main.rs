use std::process;

use rust_kernels_cuda::{layer_norm, nvfp4_quant};

fn main() {
    if let Err(err) = gpt2_bpe::run_default() {
        eprintln!("{err}");
        process::exit(1);
    }
    if let Err(err) = nvfp4_quant::run_default() {
        eprintln!("{err}");
        process::exit(1);
    }
    if let Err(err) = layer_norm::run_default() {
        eprintln!("{err}");
        process::exit(1);
    }
}
