use std::error::Error;
use std::path::PathBuf;

use cuda_core::{CudaContext, DeviceBuffer};
use gpt2_nvfp4::{GPT2_CONTEXT_LEN, GPT2_N_EMBD, HiddenState, Nvfp4Shape, TokenEmbeddingShape};
use rust_kernels_cuda::embedding::{EmbeddingArgs, EmbeddingModule};
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

const E4M3_ONE: u8 = 0x38;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn embedding_forward_decodes_token_embeddings_to_residual_only() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module = EmbeddingModule::from_module(ctx.load_module_from_file(ptx_path().as_str())?)?;

    let mut tokens = vec![0_u32; GPT2_CONTEXT_LEN];
    tokens[0] = 7;
    tokens[1] = 11;

    let mut token_embedding_bytes = vec![0_u8; TokenEmbeddingShape::BYTE_LEN];
    set_e2m1_one(&mut token_embedding_bytes, 7 * GPT2_N_EMBD);
    set_e2m1_one(&mut token_embedding_bytes, 7 * GPT2_N_EMBD + 37);
    set_e2m1_one(&mut token_embedding_bytes, 11 * GPT2_N_EMBD + 2);

    let token_embedding_scales = vec![E4M3_ONE; TokenEmbeddingShape::SCALE_LEN];
    let tokens_dev = DeviceBuffer::from_host(&stream, &tokens)?;
    let token_embedding_bytes_dev = DeviceBuffer::from_host(&stream, &token_embedding_bytes)?;
    let token_embedding_scales_dev = DeviceBuffer::from_host(&stream, &token_embedding_scales)?;

    let mut residual_dev = DeviceBuffer::<f32>::zeroed(&stream, HiddenState::LEN)?;
    let normalized_sentinel = vec![123.0_f32; HiddenState::LEN];
    let amax_sentinel = vec![456.0_f32; GPT2_CONTEXT_LEN];
    let normalized_dev = DeviceBuffer::from_host(&stream, &normalized_sentinel)?;
    let amax_dev = DeviceBuffer::from_host(&stream, &amax_sentinel)?;

    module.token_embedding(EmbeddingArgs {
        stream: &stream,
        tokens: &tokens_dev,
        token_embedding: Nvfp4DeviceTensor {
            bytes: &token_embedding_bytes_dev,
            scales: &token_embedding_scales_dev,
            global_scale: 1.0,
        },
        residual: &mut residual_dev,
        hidden_len: HiddenState::LEN as u32,
        embedding_dim: GPT2_N_EMBD as u32,
    })?;

    let residual = residual_dev.to_host_vec(&stream)?;
    let normalized = normalized_dev.to_host_vec(&stream)?;
    let amax = amax_dev.to_host_vec(&stream)?;

    assert_value(residual[0], 1.0);
    assert_value(residual[37], 1.0);
    assert_value(residual[2], 0.0);
    assert_value(residual[GPT2_N_EMBD + 2], 1.0);
    assert_value(residual[GPT2_N_EMBD], 0.0);
    assert!(normalized.iter().all(|value| *value == 123.0));
    assert!(amax.iter().all(|value| *value == 456.0));
    Ok(())
}

fn set_e2m1_one(bytes: &mut [u8], element: usize) {
    let byte = &mut bytes[element / 2];
    if element & 1 == 0 {
        *byte = (*byte & 0xf0) | 0x2;
    } else {
        *byte = (*byte & 0x0f) | 0x20;
    }
}

fn assert_value(actual: f32, expected: f32) {
    let error = (actual - expected).abs();
    assert!(
        error <= 1.0e-5,
        "actual={actual:.8e} expected={expected:.8e} error={error:.8e}"
    );
}

fn gpu_device_index() -> usize {
    std::env::var("CUDA_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn ptx_path() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../rust_kernels_cuda.ptx")
        .to_string_lossy()
        .into_owned()
}
