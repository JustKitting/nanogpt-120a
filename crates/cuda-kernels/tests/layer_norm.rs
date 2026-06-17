use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::layer_norm::{GptLayerNormArgs, LayerNormArgs, LayerNormModule, ROW_SIZE};
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

mod common;

const GPT_EMBEDDING_DIM: usize = 768;
const E2M1_ONE_PAIR: u8 = 0x22;
const E4M3_ONE: u8 = 0x38;

const SAMPLE_ROW_0: [f32; ROW_SIZE] = [
    -3.875, -3.625, -3.375, -3.125, -2.875, -2.625, -2.375, -2.125, -1.875, -1.625, -1.375, -1.125,
    -0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875, 2.125,
    2.375, 2.625, 2.875, 3.125, 3.375, 3.625, 3.875,
];

const SAMPLE_ROW_1: [f32; ROW_SIZE] = [
    -5.3125, -4.9375, -4.5625, -4.1875, -3.8125, -3.4375, -3.0625, -2.6875, -2.3125, -1.9375,
    -1.5625, -1.1875, -0.8125, -0.4375, -0.0625, 0.3125, 0.6875, 1.0625, 1.4375, 1.8125, 2.1875,
    2.5625, 2.9375, 3.3125, 3.6875, 4.0625, 4.4375, 4.8125, 5.1875, 5.5625, 5.9375, 6.3125,
];

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn layer_norm_matches_reference() -> Result<(), Box<dyn Error>> {
    let row_count = 2usize;
    let epsilon = 1.0e-5f32;
    let mut x = [0.0f32; ROW_SIZE * 2];
    let mut gamma = [0.0f32; ROW_SIZE];
    let mut beta = [0.0f32; ROW_SIZE];

    x[..ROW_SIZE].copy_from_slice(&SAMPLE_ROW_0);
    x[ROW_SIZE..].copy_from_slice(&SAMPLE_ROW_1);

    for col in 0..ROW_SIZE {
        gamma[col] = 0.75 + col as f32 * 0.01;
        beta[col] = -0.125 + col as f32 * 0.005;
    }

    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        LayerNormModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let gamma_dev = DeviceBuffer::from_host(&stream, &gamma)?;
    let beta_dev = DeviceBuffer::from_host(&stream, &beta)?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, x.len())?;

    module.layer_norm_warp_f32(LayerNormArgs {
        stream: &stream,
        x: &x_dev,
        gamma: &gamma_dev,
        beta: &beta_dev,
        out: &mut out_dev,
        row_count: row_count as u32,
        epsilon,
    })?;

    let out = out_dev.to_host_vec(&stream)?;
    let expected = reference_layer_norm(&x, &gamma, &beta, row_count, epsilon);
    let max_abs_error = max_abs_error(&out, &expected);

    assert!(max_abs_error <= 1.0e-5, "max_abs_error={max_abs_error:.8e}");
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn gpt_layer_norm_matches_reference() -> Result<(), Box<dyn Error>> {
    let row_count = 2usize;
    let epsilon = 1.0e-5f32;
    let mut x = vec![0.0f32; row_count * GPT_EMBEDDING_DIM];

    for row in 0..row_count {
        for col in 0..GPT_EMBEDDING_DIM {
            x[row * GPT_EMBEDDING_DIM + col] = (col as f32 - 383.5) * 0.001 + row as f32 * 0.25;
        }
    }

    let weight_bytes = vec![E2M1_ONE_PAIR; GPT_EMBEDDING_DIM / 2];
    let weight_scales = vec![E4M3_ONE; GPT_EMBEDDING_DIM / 16];
    let bias_bytes = vec![0_u8; GPT_EMBEDDING_DIM / 2];
    let bias_scales = vec![E4M3_ONE; GPT_EMBEDDING_DIM / 16];

    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        LayerNormModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let weight_bytes_dev = DeviceBuffer::from_host(&stream, &weight_bytes)?;
    let weight_scales_dev = DeviceBuffer::from_host(&stream, &weight_scales)?;
    let bias_bytes_dev = DeviceBuffer::from_host(&stream, &bias_bytes)?;
    let bias_scales_dev = DeviceBuffer::from_host(&stream, &bias_scales)?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, x.len())?;
    let mut amax_dev = DeviceBuffer::<f32>::zeroed(&stream, row_count)?;
    let mut mean_dev = DeviceBuffer::<f32>::zeroed(&stream, row_count)?;
    let mut inv_std_dev = DeviceBuffer::<f32>::zeroed(&stream, row_count)?;

    module.gpt_layer_norm(GptLayerNormArgs {
        stream: &stream,
        residual: &x_dev,
        weight: Nvfp4DeviceTensor {
            bytes: &weight_bytes_dev,
            scales: &weight_scales_dev,
            global_scale: 1.0,
        },
        bias: Nvfp4DeviceTensor {
            bytes: &bias_bytes_dev,
            scales: &bias_scales_dev,
            global_scale: 1.0,
        },
        normalized: &mut out_dev,
        normalized_amax: &mut amax_dev,
        mean: &mut mean_dev,
        inv_std: &mut inv_std_dev,
        row_count: row_count as u32,
        embedding_dim: GPT_EMBEDDING_DIM as u32,
        epsilon,
    })?;

    let out = out_dev.to_host_vec(&stream)?;
    let amax = amax_dev.to_host_vec(&stream)?;
    let expected = reference_layer_norm_rows(&x, row_count, GPT_EMBEDDING_DIM, epsilon);
    let max_abs_error = max_abs_error(&out, &expected);

    assert!(max_abs_error <= 1.0e-4, "max_abs_error={max_abs_error:.8e}");
    assert_row_amax(&out, &amax, row_count, GPT_EMBEDDING_DIM);
    Ok(())
}

fn reference_layer_norm(
    x: &[f32; ROW_SIZE * 2],
    gamma: &[f32; ROW_SIZE],
    beta: &[f32; ROW_SIZE],
    row_count: usize,
    epsilon: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; row_count * ROW_SIZE];
    for row in 0..row_count {
        let base = row * ROW_SIZE;
        let mean = x[base..base + ROW_SIZE].iter().sum::<f32>() / ROW_SIZE as f32;
        let variance = x[base..base + ROW_SIZE]
            .iter()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f32>()
            / ROW_SIZE as f32;
        let inv_std = 1.0 / (variance + epsilon).sqrt();

        for col in 0..ROW_SIZE {
            let centered = x[base + col] - mean;
            out[base + col] = (centered * inv_std).mul_add(gamma[col], beta[col]);
        }
    }
    out
}

fn reference_layer_norm_rows(
    x: &[f32],
    row_count: usize,
    row_len: usize,
    epsilon: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; row_count * row_len];
    for row in 0..row_count {
        let base = row * row_len;
        let mean = x[base..base + row_len].iter().sum::<f32>() / row_len as f32;
        let variance = x[base..base + row_len]
            .iter()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f32>()
            / row_len as f32;
        let inv_std = 1.0 / (variance + epsilon).sqrt();

        for col in 0..row_len {
            out[base + col] = (x[base + col] - mean) * inv_std;
        }
    }
    out
}

fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected.iter())
        .fold(0.0f32, |max, (actual, expected)| {
            max.max((actual - expected).abs())
        })
}

fn assert_row_amax(out: &[f32], amax: &[f32], row_count: usize, row_len: usize) {
    for (row, actual) in amax.iter().copied().enumerate().take(row_count) {
        let base = row * row_len;
        let expected = out[base..base + row_len]
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        let error = (actual - expected).abs();
        assert!(error <= 1.0e-5, "row={row} error={error:.8e}");
    }
}
