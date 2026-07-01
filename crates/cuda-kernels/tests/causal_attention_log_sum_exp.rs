use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer};
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
    let (_, stream, module) = common::cuda_test_module(AttentionModule::from_module)?;
    let run = run_attention(&stream, &module, vec![0.0_f32; TOKEN_COUNT * QKV_DIM], 1)?;

    assert!(run.out.iter().all(|value| value.abs() <= TOLERANCE));

    for head in 0..HEAD_COUNT {
        let base = head * TOKEN_COUNT;
        assert!(run.log_sum_exp[base].abs() <= TOLERANCE);
        for token in 1..TOKEN_COUNT {
            assert!(run.log_sum_exp[base + token] > run.log_sum_exp[base + token - 1]);
        }
    }

    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn causal_attention_batch_isolation() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(AttentionModule::from_module)?;

    let first = run_attention(&stream, &module, sample_qkv(0.25), 2)?;
    let second = run_attention(&stream, &module, sample_qkv(8.0), 2)?;

    assert_eq!(bits(&first.out[..TOKEN_COUNT * EMBEDDING_DIM]), bits(&second.out[..TOKEN_COUNT * EMBEDDING_DIM]));
    assert_eq!(
        bits(&first.log_sum_exp[..HEAD_COUNT * TOKEN_COUNT]),
        bits(&second.log_sum_exp[..HEAD_COUNT * TOKEN_COUNT])
    );

    Ok(())
}

struct AttentionRun {
    out: Vec<f32>,
    log_sum_exp: Vec<f32>,
}

fn run_attention(
    stream: &CudaStream,
    module: &AttentionModule,
    qkv_values: Vec<f32>,
    batch_size: usize,
) -> Result<AttentionRun, Box<dyn Error>> {
    let qkv = DeviceBuffer::from_host(stream, &qkv_values)?;
    let mut out = DeviceBuffer::<f32>::zeroed(stream, batch_size * TOKEN_COUNT * EMBEDDING_DIM)?;
    let mut log_sum_exp =
        DeviceBuffer::<f32>::zeroed(stream, batch_size * HEAD_COUNT * TOKEN_COUNT)?;

    module.causal_attention(CausalAttentionArgs {
        stream,
        qkv: &qkv,
        out: &mut out,
        log_sum_exp: &mut log_sum_exp,
        row_count: (batch_size * TOKEN_COUNT) as u32,
        seq_len: TOKEN_COUNT as u32,
        batch_size: batch_size as u32,
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
    let mut qkv = vec![0.0_f32; 2 * TOKEN_COUNT * QKV_DIM];
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

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}
