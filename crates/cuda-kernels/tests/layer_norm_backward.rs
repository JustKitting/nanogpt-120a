use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::layer_norm_backward::{LayerNormBackwardInputArgs, LayerNormBackwardModule};
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

mod common;

const ROWS: usize = 2;
const COLS: usize = 32;
const E2M1_ONE_PAIR: u8 = 0x22;
const E4M3_ONE: u8 = 0x38;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn layer_norm_backward_input_matches_reference() -> Result<(), Box<dyn Error>> {
    let epsilon = 1.0e-5f32;
    let x = sample_residual();
    let d_normalized = sample_grad();
    let (mean, inv_std) = reference_stats(&x, epsilon);
    let weight_bytes = vec![E2M1_ONE_PAIR; COLS / 2];
    let weight_scales = vec![E4M3_ONE; COLS / 16];

    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module = LayerNormBackwardModule::from_module(
        ctx.load_module_from_file(common::ptx_path().as_str())?,
    )?;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let grad_dev = DeviceBuffer::from_host(&stream, &d_normalized)?;
    let mean_dev = DeviceBuffer::from_host(&stream, &mean)?;
    let inv_std_dev = DeviceBuffer::from_host(&stream, &inv_std)?;
    let weight_bytes_dev = DeviceBuffer::from_host(&stream, &weight_bytes)?;
    let weight_scales_dev = DeviceBuffer::from_host(&stream, &weight_scales)?;
    let mut dx_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS * COLS)?;

    module.backward_input(LayerNormBackwardInputArgs {
        stream: &stream,
        residual: &x_dev,
        d_normalized: &grad_dev,
        mean: &mean_dev,
        inv_std: &inv_std_dev,
        weight: Nvfp4DeviceTensor {
            bytes: &weight_bytes_dev,
            scales: &weight_scales_dev,
            global_scale: 1.0,
        },
        d_residual: &mut dx_dev,
        row_count: ROWS as u32,
        embedding_dim: COLS as u32,
    })?;

    let dx = dx_dev.to_host_vec(&stream)?;
    let expected = reference_backward_input(&x, &d_normalized, &mean, &inv_std);
    let error = max_abs_error(&dx, &expected);
    assert!(error <= 1.0e-4, "max_abs_error={error:.8e}");
    Ok(())
}

fn sample_residual() -> Vec<f32> {
    (0..ROWS * COLS)
        .map(|i| (i as f32 % 17.0 - 8.0) * 0.125 + (i / COLS) as f32 * 0.25)
        .collect()
}

fn sample_grad() -> Vec<f32> {
    (0..ROWS * COLS)
        .map(|i| (i as f32 % 11.0 - 5.0) * 0.03125)
        .collect()
}

fn reference_stats(x: &[f32], epsilon: f32) -> (Vec<f32>, Vec<f32>) {
    let mut mean = vec![0.0f32; ROWS];
    let mut inv_std = vec![0.0f32; ROWS];
    for row in 0..ROWS {
        let base = row * COLS;
        mean[row] = x[base..base + COLS].iter().sum::<f32>() / COLS as f32;
        let centered = x[base..base + COLS]
            .iter()
            .map(|value| value - mean[row])
            .collect::<Vec<_>>();
        let variance = centered.iter().map(|value| value * value).sum::<f32>() / COLS as f32;
        inv_std[row] = 1.0 / (variance + epsilon).sqrt();
    }
    (mean, inv_std)
}

fn reference_backward_input(x: &[f32], grad: &[f32], mean: &[f32], inv_std: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; ROWS * COLS];
    for row in 0..ROWS {
        let base = row * COLS;
        let xhat = x[base..base + COLS]
            .iter()
            .map(|value| (value - mean[row]) * inv_std[row])
            .collect::<Vec<_>>();
        let sum_grad = grad[base..base + COLS].iter().sum::<f32>();
        let sum_xhat_grad = xhat
            .iter()
            .zip(&grad[base..base + COLS])
            .map(|(xhat, grad)| xhat * grad)
            .sum::<f32>();
        for col in 0..COLS {
            out[base + col] = (grad[base + col]
                - sum_grad / COLS as f32
                - xhat[col] * sum_xhat_grad / COLS as f32)
                * inv_std[row];
        }
    }
    out
}

fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .fold(0.0, |max, (a, e)| max.max((a - e).abs()))
}
