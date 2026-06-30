use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::lm_head::{LmHeadArgs, LmHeadModule};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

mod common;

use common::set_e2m1_one;

const TOKEN_COUNT: usize = 2;
const INPUT_DIM: usize = 64;
const VOCAB_SIZE: usize = 16;
const E4M3_ONE: u8 = 0x38;
const TOLERANCE: f32 = 1.0e-7;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn lm_head_projects_rowwise_nvfp4_hidden_to_logits() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        LmHeadModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;

    let mut input_bytes = vec![0_u8; TOKEN_COUNT * INPUT_DIM / 2];
    set_e2m1_one(&mut input_bytes, 0);
    set_e2m1_one(&mut input_bytes, 1);
    set_e2m1_one(&mut input_bytes, INPUT_DIM + 2);
    let input_scales = vec![E4M3_ONE; TOKEN_COUNT * INPUT_DIM / 16];
    let input_global_scales = vec![1.0_f32; TOKEN_COUNT];

    let mut weight_bytes = vec![0_u8; VOCAB_SIZE * INPUT_DIM / 2];
    set_e2m1_one(&mut weight_bytes, 0);
    set_e2m1_one(&mut weight_bytes, INPUT_DIM + 1);
    set_e2m1_one(&mut weight_bytes, 2 * INPUT_DIM + 2);
    let weight_scales = vec![E4M3_ONE; VOCAB_SIZE * INPUT_DIM / 16];

    let input_bytes_dev = DeviceBuffer::from_host(&stream, &input_bytes)?;
    let input_scales_dev = DeviceBuffer::from_host(&stream, &input_scales)?;
    let input_global_scales_dev = DeviceBuffer::from_host(&stream, &input_global_scales)?;
    let weight_bytes_dev = DeviceBuffer::from_host(&stream, &weight_bytes)?;
    let weight_scales_dev = DeviceBuffer::from_host(&stream, &weight_scales)?;
    let weight_global_scale_dev = DeviceBuffer::from_host(&stream, &[1.0_f32])?;
    let mut logits_dev = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * VOCAB_SIZE)?;

    module.logits(LmHeadArgs {
        stream: &stream,
        input: Nvfp4RowwiseDeviceTensor {
            bytes: &input_bytes_dev,
            scales: &input_scales_dev,
            global_scales: &input_global_scales_dev,
        },
        weight: Nvfp4FourSixMmaWeightTensor {
            bytes: &weight_bytes_dev,
            scales: &weight_scales_dev,
            global_scale: &weight_global_scale_dev,
        },
        logits: &mut logits_dev,
        token_count: TOKEN_COUNT as u32,
        input_dim: INPUT_DIM as u32,
        vocab_size: VOCAB_SIZE as u32,
    })?;

    let logits = logits_dev.to_host_vec(&stream)?;
    assert_value(logits[0], 1.0);
    assert_value(logits[1], 1.0);
    assert_value(logits[2], 0.0);
    assert_value(logits[VOCAB_SIZE], 0.0);
    assert_value(logits[VOCAB_SIZE + 1], 0.0);
    assert_value(logits[VOCAB_SIZE + 2], 1.0);
    Ok(())
}

fn assert_value(actual: f32, expected: f32) {
    let error = (actual - expected).abs();
    assert!(
        error <= TOLERANCE,
        "actual={actual:.8e} expected={expected:.8e} error={error:.8e}"
    );
}
