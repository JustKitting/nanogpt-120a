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
fn causal_attention_writes_log_sum_exp() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        AttentionModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;

    let qkv = DeviceBuffer::from_host(&stream, &vec![0.0_f32; TOKEN_COUNT * QKV_DIM])?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * EMBEDDING_DIM)?;
    let mut log_sum_exp = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * HEAD_COUNT)?;

    module.causal_attention(CausalAttentionArgs {
        stream: &stream,
        qkv: &qkv,
        out: &mut out,
        log_sum_exp: &mut log_sum_exp,
        row_count: TOKEN_COUNT as u32,
        seq_len: TOKEN_COUNT as u32,
        batch_size: 1,
        embedding_dim: EMBEDDING_DIM as u32,
        qkv_dim: QKV_DIM as u32,
        head_count: HEAD_COUNT as u32,
        head_dim: HEAD_DIM as u32,
    })?;

    let actual_out = out.to_host_vec(&stream)?;
    let actual_log_sum_exp = log_sum_exp.to_host_vec(&stream)?;
    assert!(actual_out.iter().all(|value| value.abs() <= TOLERANCE));

    for head in 0..HEAD_COUNT {
        let base = head * TOKEN_COUNT;
        assert!(actual_log_sum_exp[base].abs() <= TOLERANCE);
        for token in 1..TOKEN_COUNT {
            assert!(actual_log_sum_exp[base + token] > actual_log_sum_exp[base + token - 1]);
        }
    }

    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn causal_attention_batch_isolation() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        AttentionModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;

    let first = run_batched_attention(&stream, &module, sample_qkv(0.25))?;
    let second = run_batched_attention(&stream, &module, sample_qkv(8.0))?;

    let sample0_out_len = TOKEN_COUNT * EMBEDDING_DIM;
    let sample0_log_sum_exp_len = HEAD_COUNT * TOKEN_COUNT;
    assert_eq!(
        first.out[..sample0_out_len]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        second.out[..sample0_out_len]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        first.log_sum_exp[..sample0_log_sum_exp_len]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        second.log_sum_exp[..sample0_log_sum_exp_len]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    Ok(())
}

struct AttentionRun {
    out: Vec<f32>,
    log_sum_exp: Vec<f32>,
}

fn run_batched_attention(
    stream: &cuda_core::CudaStream,
    module: &AttentionModule,
    qkv_values: Vec<f32>,
) -> Result<AttentionRun, Box<dyn Error>> {
    const BATCH_SIZE: usize = 2;
    let qkv = DeviceBuffer::from_host(stream, &qkv_values)?;
    let mut out = DeviceBuffer::<f32>::zeroed(stream, BATCH_SIZE * TOKEN_COUNT * EMBEDDING_DIM)?;
    let mut log_sum_exp =
        DeviceBuffer::<f32>::zeroed(stream, BATCH_SIZE * HEAD_COUNT * TOKEN_COUNT)?;

    module.causal_attention(CausalAttentionArgs {
        stream,
        qkv: &qkv,
        out: &mut out,
        log_sum_exp: &mut log_sum_exp,
        row_count: (BATCH_SIZE * TOKEN_COUNT) as u32,
        seq_len: TOKEN_COUNT as u32,
        batch_size: BATCH_SIZE as u32,
        embedding_dim: EMBEDDING_DIM as u32,
        qkv_dim: QKV_DIM as u32,
        head_count: HEAD_COUNT as u32,
        head_dim: HEAD_DIM as u32,
    })?;

    Ok(AttentionRun {
        out: out.to_host_vec(stream)?,
        log_sum_exp: log_sum_exp.to_host_vec(stream)?,
    })
}

fn sample_qkv(sample1_scale: f32) -> Vec<f32> {
    const BATCH_SIZE: usize = 2;
    let mut qkv = vec![0.0_f32; BATCH_SIZE * TOKEN_COUNT * QKV_DIM];
    fill_sample(&mut qkv, 0, 1.0);
    fill_sample(&mut qkv, 1, sample1_scale);
    qkv
}

fn fill_sample(qkv: &mut [f32], batch: usize, scale: f32) {
    for token in 0..TOKEN_COUNT {
        for head in 0..HEAD_COUNT {
            for dim in 0..HEAD_DIM {
                let row_base = (batch * TOKEN_COUNT + token) * QKV_DIM;
                let head_dim = head * HEAD_DIM + dim;
                let value = scale * (1.0 + token as f32 * 0.125 + head as f32 * 0.25);
                qkv[row_base + head_dim] = value + dim as f32 * 0.01;
                qkv[row_base + EMBEDDING_DIM + head_dim] = value * 0.5 + dim as f32 * 0.02;
                qkv[row_base + 2 * EMBEDDING_DIM + head_dim] = value * 0.25 + dim as f32 * 0.03;
            }
        }
    }
}
