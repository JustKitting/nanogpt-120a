use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::attention::{AttentionModule, CausalAttentionArgs};

mod common;

const TOKEN_COUNT: usize = 4;
const HEAD_COUNT: usize = 2;
const HEAD_DIM: usize = 2;
const EMBEDDING_DIM: usize = HEAD_COUNT * HEAD_DIM;
const QKV_DIM: usize = 3 * EMBEDDING_DIM;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn causal_attention_matches_reference() -> Result<(), Box<dyn Error>> {
    let qkv = sample_qkv();

    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        AttentionModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;

    let qkv_dev = DeviceBuffer::from_host(&stream, &qkv)?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * EMBEDDING_DIM)?;

    module.causal_attention(CausalAttentionArgs {
        stream: &stream,
        qkv: &qkv_dev,
        out: &mut out_dev,
        token_count: TOKEN_COUNT as u32,
        embedding_dim: EMBEDDING_DIM as u32,
        qkv_dim: QKV_DIM as u32,
        head_count: HEAD_COUNT as u32,
        head_dim: HEAD_DIM as u32,
    })?;

    let out = out_dev.to_host_vec(&stream)?;
    let expected = reference_causal_attention(&qkv);
    let max_abs_error = max_abs_error(&out, &expected);

    assert!(max_abs_error <= 1.0e-3, "max_abs_error={max_abs_error:.8e}");
    Ok(())
}

fn sample_qkv() -> [f32; TOKEN_COUNT * QKV_DIM] {
    let mut qkv = [0.0f32; TOKEN_COUNT * QKV_DIM];

    for token in 0..TOKEN_COUNT {
        for head in 0..HEAD_COUNT {
            for dim in 0..HEAD_DIM {
                let base = token as f32 * 0.17 + head as f32 * 0.11 + dim as f32 * 0.07;
                qkv[qkv_index(token, head, dim, 0)] = 0.25 + base;
                qkv[qkv_index(token, head, dim, EMBEDDING_DIM)] = -0.15 + base * 0.5;
                qkv[qkv_index(token, head, dim, EMBEDDING_DIM * 2)] = 0.4 - base * 0.25;
            }
        }
    }

    qkv
}

fn reference_causal_attention(qkv: &[f32; TOKEN_COUNT * QKV_DIM]) -> Vec<f32> {
    let mut out = vec![0.0f32; TOKEN_COUNT * EMBEDDING_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    for query in 0..TOKEN_COUNT {
        for head in 0..HEAD_COUNT {
            let mut scores = [0.0f32; TOKEN_COUNT];
            for key in 0..=query {
                let mut dot = 0.0f32;
                for dim in 0..HEAD_DIM {
                    dot += qkv[qkv_index(query, head, dim, 0)]
                        * qkv[qkv_index(key, head, dim, EMBEDDING_DIM)];
                }
                scores[key] = dot * scale;
            }

            let score_max = scores[..=query]
                .iter()
                .fold(f32::NEG_INFINITY, |max, score| max.max(*score));
            let denom = scores[..=query]
                .iter()
                .map(|score| (score - score_max).exp())
                .sum::<f32>();

            for dim in 0..HEAD_DIM {
                let mut value = 0.0f32;
                for key in 0..=query {
                    let weight = (scores[key] - score_max).exp() / denom;
                    value += weight * qkv[qkv_index(key, head, dim, EMBEDDING_DIM * 2)];
                }
                out[query * EMBEDDING_DIM + head * HEAD_DIM + dim] = value;
            }
        }
    }

    out
}

fn qkv_index(token: usize, head: usize, dim: usize, section_offset: usize) -> usize {
    token * QKV_DIM + section_offset + head * HEAD_DIM + dim
}

fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected.iter())
        .fold(0.0f32, |max, (actual, expected)| {
            max.max((actual - expected).abs())
        })
}
