use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::attention::{AttentionModule, CausalAttentionArgs};

mod common;

const TOKEN_COUNT: usize = 4;
const HEAD_COUNT: usize = 2;
const HEAD_DIM: usize = 4;
const EMBEDDING_DIM: usize = HEAD_COUNT * HEAD_DIM;
const QKV_DIM: usize = EMBEDDING_DIM * 3;
const TOLERANCE: f32 = 1.0e-7;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn causal_attention_writes_lse() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        AttentionModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;

    let qkv = DeviceBuffer::from_host(&stream, &vec![0.0_f32; TOKEN_COUNT * QKV_DIM])?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * EMBEDDING_DIM)?;
    let mut lse = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * HEAD_COUNT)?;

    module.causal_attention(CausalAttentionArgs {
        stream: &stream,
        qkv: &qkv,
        out: &mut out,
        lse: &mut lse,
        token_count: TOKEN_COUNT as u32,
        embedding_dim: EMBEDDING_DIM as u32,
        qkv_dim: QKV_DIM as u32,
        head_count: HEAD_COUNT as u32,
        head_dim: HEAD_DIM as u32,
    })?;

    let actual_out = out.to_host_vec(&stream)?;
    let actual_lse = lse.to_host_vec(&stream)?;
    assert!(actual_out.iter().all(|value| value.abs() <= TOLERANCE));

    for head in 0..HEAD_COUNT {
        let base = head * TOKEN_COUNT;
        assert!(actual_lse[base].abs() <= TOLERANCE);
        for token in 1..TOKEN_COUNT {
            assert!(actual_lse[base + token] > actual_lse[base + token - 1]);
        }
    }

    Ok(())
}
