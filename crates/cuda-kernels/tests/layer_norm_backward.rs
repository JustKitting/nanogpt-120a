use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::layer_norm_backward::{
    LayerNormBackwardInputF32Args, LayerNormBackwardModule,
};
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

mod common;
#[path = "layer_norm/stats.rs"]
mod stats;

use common::max_abs_error;
use common::nvfp4::{one_pair_bytes, one_scales};
use stats::{reference_row_stats, sample_rows};

const ROWS: usize = 2;
const COLS: usize = 32;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn layer_norm_backward_input_matches_reference() -> Result<(), Box<dyn Error>> {
    let epsilon = 1.0e-5f32;
    let x = sample_residual();
    let d_normalized = sample_grad();
    let (mean, inv_std) = reference_row_stats(&x, ROWS, COLS, epsilon);

    let (_, stream, module) = common::cuda_test_module(LayerNormBackwardModule::from_module)?;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let grad_dev = DeviceBuffer::from_host(&stream, &d_normalized)?;
    let mean_dev = DeviceBuffer::from_host(&stream, &mean)?;
    let inv_std_dev = DeviceBuffer::from_host(&stream, &inv_std)?;
    let weight_bytes_dev = DeviceBuffer::from_host(&stream, &one_pair_bytes(COLS))?;
    let weight_scales_dev = DeviceBuffer::from_host(&stream, &one_scales(COLS))?;
    let weight_global_scale_dev = DeviceBuffer::from_host(&stream, &[1.0_f32])?;
    let mut dx_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS * COLS)?;

    module.backward_input_f32(LayerNormBackwardInputF32Args {
        stream: &stream,
        residual: &x_dev,
        d_normalized: &grad_dev,
        mean: &mean_dev,
        inv_std: &inv_std_dev,
        weight: Nvfp4DeviceTensor::new(
            &weight_bytes_dev,
            &weight_scales_dev,
            &weight_global_scale_dev,
        ),
        d_residual: &mut dx_dev,
        row_count: ROWS as u32,
        embedding_dim: COLS as u32,
    })?;

    let dx = dx_dev.to_host_vec(&stream)?;
    let expected = reference_backward_input(&x, &d_normalized, &mean, &inv_std);
    let error = max_abs_error(&dx, &expected);
    assert!(error <= 1.0e-8, "max_abs_error={error:.8e}");
    Ok(())
}

fn sample_residual() -> Vec<f32> {
    sample_rows(ROWS, COLS, 17, 8.0, 0.125, 0.25)
}

fn sample_grad() -> Vec<f32> {
    sample_rows(ROWS, COLS, 11, 5.0, 0.03125, 0.0)
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
