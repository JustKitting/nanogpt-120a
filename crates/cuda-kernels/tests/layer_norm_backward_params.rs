use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::layer_norm_backward::{
    LayerNormBackwardModule, LayerNormBackwardParamF32Args,
};

mod common;
#[path = "layer_norm/stats.rs"]
mod stats;

use stats::reference_row_stats;

const ROWS: usize = 3;
const COLS: usize = 32;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn layer_norm_backward_params_match_reference() -> Result<(), Box<dyn Error>> {
    let epsilon = 1.0e-5f32;
    let x = sample_residual();
    let dy = sample_grad();
    let (mean, inv_std) = reference_row_stats(&x, ROWS, COLS, epsilon);

    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = LayerNormBackwardModule::from_module(ptx)?;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
    let mean_dev = DeviceBuffer::from_host(&stream, &mean)?;
    let inv_std_dev = DeviceBuffer::from_host(&stream, &inv_std)?;
    let mut d_weight_dev = DeviceBuffer::<f32>::zeroed(&stream, COLS)?;
    let mut d_bias_dev = DeviceBuffer::<f32>::zeroed(&stream, COLS)?;

    module.backward_params_f32(LayerNormBackwardParamF32Args {
        stream: &stream,
        residual: &x_dev,
        d_normalized: &dy_dev,
        mean: &mean_dev,
        inv_std: &inv_std_dev,
        d_weight: &mut d_weight_dev,
        d_bias: &mut d_bias_dev,
        row_count: ROWS as u32,
        embedding_dim: COLS as u32,
    })?;

    let d_weight = d_weight_dev.to_host_vec(&stream)?;
    let d_bias = d_bias_dev.to_host_vec(&stream)?;
    let (expected_weight, expected_bias) = reference_param_grads(&x, &dy, &mean, &inv_std);
    common::assert_slice_close(&d_weight, &expected_weight, 1.0e-7);
    common::assert_slice_close(&d_bias, &expected_bias, 1.0e-7);
    Ok(())
}

fn sample_residual() -> Vec<f32> {
    (0..ROWS * COLS)
        .map(|i| (i as f32 % 19.0 - 9.0) * 0.125 + (i / COLS) as f32 * 0.25)
        .collect()
}

fn sample_grad() -> Vec<f32> {
    (0..ROWS * COLS)
        .map(|i| (i as f32 % 13.0 - 6.0) * 0.03125)
        .collect()
}

fn reference_param_grads(
    x: &[f32],
    dy: &[f32],
    mean: &[f32],
    inv_std: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut d_weight = vec![0.0f32; COLS];
    let mut d_bias = vec![0.0f32; COLS];
    for row in 0..ROWS {
        for col in 0..COLS {
            let offset = row * COLS + col;
            let xhat = (x[offset] - mean[row]) * inv_std[row];
            d_weight[col] += dy[offset] * xhat;
            d_bias[col] += dy[offset];
        }
    }
    (d_weight, d_bias)
}
