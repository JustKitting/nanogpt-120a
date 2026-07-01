use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::layer_norm::{GptLayerNormArgs, LayerNormArgs, LayerNormModule, ROW_SIZE};
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

mod common;
#[path = "layer_norm/reference.rs"]
mod reference;

use common::max_abs_error;
use common::nvfp4::{one_pair_bytes, one_scales};
use reference::{
    assert_row_amax, reference_layer_norm, reference_layer_norm_rows, sample_row_0, sample_row_1,
};

const GPT_EMBEDDING_DIM: usize = 768;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn layer_norm_matches_reference() -> Result<(), Box<dyn Error>> {
    let row_count = 2usize;
    let epsilon = 1.0e-5f32;
    let mut x = [0.0f32; ROW_SIZE * 2];
    let mut gamma = [0.0f32; ROW_SIZE];
    let mut beta = [0.0f32; ROW_SIZE];

    for col in 0..ROW_SIZE {
        x[col] = sample_row_0(col);
        x[ROW_SIZE + col] = sample_row_1(col);
        gamma[col] = 0.75 + col as f32 * 0.01;
        beta[col] = -0.125 + col as f32 * 0.005;
    }

    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = LayerNormModule::from_module(ptx)?;

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

    assert!(max_abs_error <= 1.0e-9, "max_abs_error={max_abs_error:.8e}");
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

    let bias_bytes = vec![0_u8; GPT_EMBEDDING_DIM / 2];

    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = LayerNormModule::from_module(ptx)?;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let weight_bytes_dev = DeviceBuffer::from_host(&stream, &one_pair_bytes(GPT_EMBEDDING_DIM))?;
    let weight_scales_dev = DeviceBuffer::from_host(&stream, &one_scales(GPT_EMBEDDING_DIM))?;
    let bias_bytes_dev = DeviceBuffer::from_host(&stream, &bias_bytes)?;
    let bias_scales_dev = DeviceBuffer::from_host(&stream, &one_scales(GPT_EMBEDDING_DIM))?;
    let weight_global_scale_dev = DeviceBuffer::from_host(&stream, &[1.0_f32])?;
    let bias_global_scale_dev = DeviceBuffer::from_host(&stream, &[1.0_f32])?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, x.len())?;
    let mut amax_dev = DeviceBuffer::<f32>::zeroed(&stream, row_count)?;
    let mut mean_dev = DeviceBuffer::<f32>::zeroed(&stream, row_count)?;
    let mut inv_std_dev = DeviceBuffer::<f32>::zeroed(&stream, row_count)?;

    module.gpt_layer_norm(GptLayerNormArgs {
        stream: &stream,
        residual: &x_dev,
        weight: Nvfp4DeviceTensor::new(
            &weight_bytes_dev,
            &weight_scales_dev,
            &weight_global_scale_dev,
        ),
        bias: Nvfp4DeviceTensor::new(&bias_bytes_dev, &bias_scales_dev, &bias_global_scale_dev),
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

    assert!(max_abs_error <= 1.0e-7, "max_abs_error={max_abs_error:.8e}");
    assert_row_amax(&out, &amax, row_count, GPT_EMBEDDING_DIM);
    Ok(())
}
