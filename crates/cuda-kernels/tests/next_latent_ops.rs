use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::next_latent::{
    NextLatGeluArgs, NextLatModule, NextLatProjectionArgs, NextLatResidualAddArgs,
};
use rust_kernels_cuda::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};

mod common;

use common::nvfp4::{E2M1_ONE_PAIR, E4M3_ONE};

const TOKEN_COUNT: usize = 32;
const INPUT_DIM: usize = 128;
const OUTPUT_DIM: usize = 64;
const TOLERANCE: f32 = 1.0e-6;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nextlat_projection_gelu_and_residual_match_reference() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = NextLatModule::from_module(ptx)?;

    let zeros = vec![0_u8; TOKEN_COUNT * INPUT_DIM / 2];
    let scales = vec![E4M3_ONE; TOKEN_COUNT * INPUT_DIM / 16];
    let globals = vec![1.0_f32; TOKEN_COUNT];
    let weight_zeros = vec![0_u8; INPUT_DIM * OUTPUT_DIM / 2];
    let weight_scales = vec![E4M3_ONE; INPUT_DIM * OUTPUT_DIM / 16];
    let bias = vec![E2M1_ONE_PAIR; OUTPUT_DIM / 2];
    let bias_scales = vec![E4M3_ONE; OUTPUT_DIM / 16];
    let global_one = [1.0_f32];

    let input_bytes = DeviceBuffer::from_host(&stream, &zeros)?;
    let input_scales = DeviceBuffer::from_host(&stream, &scales)?;
    let input_globals = DeviceBuffer::from_host(&stream, &globals)?;
    let weight_bytes = DeviceBuffer::from_host(&stream, &weight_zeros)?;
    let weight_scales_dev = DeviceBuffer::from_host(&stream, &weight_scales)?;
    let bias_bytes = DeviceBuffer::from_host(&stream, &bias)?;
    let bias_scales_dev = DeviceBuffer::from_host(&stream, &bias_scales)?;
    let global_dev = DeviceBuffer::from_host(&stream, &global_one)?;
    let mut projection = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * OUTPUT_DIM)?;
    let mut gelu = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * OUTPUT_DIM)?;
    let residual = DeviceBuffer::from_host(&stream, &vec![0.25_f32; TOKEN_COUNT * OUTPUT_DIM])?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * OUTPUT_DIM)?;

    module.projection(NextLatProjectionArgs {
        stream: &stream,
        input: Nvfp4RowwiseDeviceTensor::new(&input_bytes, &input_scales, &input_globals),
        weight: Nvfp4FourSixMmaWeightTensor::new(&weight_bytes, &weight_scales_dev, &global_dev),
        bias: Nvfp4DeviceTensor::new(&bias_bytes, &bias_scales_dev, &global_dev),
        out: &mut projection,
        token_count: TOKEN_COUNT as u32,
        input_dim: INPUT_DIM as u32,
        output_dim: OUTPUT_DIM as u32,
    })?;
    module.gelu(NextLatGeluArgs {
        stream: &stream,
        input: &projection,
        out: &mut gelu,
        len: (TOKEN_COUNT * OUTPUT_DIM) as u32,
    })?;
    module.residual_add(NextLatResidualAddArgs {
        stream: &stream,
        delta: &gelu,
        residual: &residual,
        out: &mut out,
        len: (TOKEN_COUNT * OUTPUT_DIM) as u32,
    })?;

    let projection = projection.to_host_vec(&stream)?;
    let gelu = gelu.to_host_vec(&stream)?;
    let out = out.to_host_vec(&stream)?;
    let expected_gelu = reference_gelu(1.0);

    assert_all_close(&projection, 1.0);
    assert_all_close(&gelu, expected_gelu);
    assert_all_close(&out, expected_gelu + 0.25);
    Ok(())
}

fn reference_gelu(x: f32) -> f32 {
    let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

fn assert_all_close(actual: &[f32], expected: f32) {
    for (index, actual) in actual.iter().enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error <= TOLERANCE,
            "index={index} actual={actual} expected={expected}"
        );
    }
}
