#![allow(dead_code)]

use std::{path::PathBuf, sync::Arc};

use cuda_core::{CudaContext, CudaModule, CudaStream, DriverError};
use gpt2_nvfp4::{AttentionLogSumExp, GPT2_BATCH_SIZE, GPT2_N_HEAD, GPT2_SEQ_LEN, GPT2_TOKEN_ROWS};

pub mod f16;
#[path = "../support/forward_scratch.rs"]
pub mod forward_scratch;
pub mod nvfp4;
pub mod saved_block;
pub mod upload;

pub type CudaTestContext = (Arc<CudaContext>, Arc<CudaStream>, Arc<CudaModule>);

pub fn cuda_test_context() -> Result<CudaTestContext, DriverError> {
    let device_index = std::env::var("CUDA_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let ptx_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../rust_kernels_cuda.ptx")
        .to_string_lossy()
        .into_owned();
    let ctx = CudaContext::new(device_index)?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(ptx_path.as_str())?;
    Ok((ctx, stream, ptx))
}

pub fn assert_nonzero_finite(values: &[f32]) {
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().any(|value| value.abs() > 0.0));
}

pub fn float_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

pub fn row_ones() -> Vec<f32> {
    vec![1.0; GPT2_TOKEN_ROWS]
}

pub fn attention_log_sum_exp_values() -> Vec<f32> {
    let mut log_sum_exp = vec![0.0_f32; AttentionLogSumExp::LEN];
    for batch in 0..GPT2_BATCH_SIZE {
        for head in 0..GPT2_N_HEAD {
            for token in 0..GPT2_SEQ_LEN {
                let index = (batch * GPT2_N_HEAD + head) * GPT2_SEQ_LEN + token;
                log_sum_exp[index] = ((token + 1) as f32).ln();
            }
        }
    }
    log_sum_exp
}
