use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::lm_head::{LmHeadArgs, LmHeadModule};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

mod common;

use common::nvfp4::{one_scales, set_e2m1_one};

const TOKEN_COUNT: usize = 2;
const INPUT_DIM: usize = 64;
const VOCAB_SIZE: usize = 16;
const TOLERANCE: f32 = 1.0e-7;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn lm_head_projects_rowwise_nvfp4_hidden_to_logits() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = LmHeadModule::from_module(ptx)?;

    let mut input_bytes = vec![0_u8; TOKEN_COUNT * INPUT_DIM / 2];
    set_e2m1_one(&mut input_bytes, 0);
    set_e2m1_one(&mut input_bytes, 1);
    set_e2m1_one(&mut input_bytes, INPUT_DIM + 2);

    let mut weight_bytes = vec![0_u8; VOCAB_SIZE * INPUT_DIM / 2];
    set_e2m1_one(&mut weight_bytes, 0);
    set_e2m1_one(&mut weight_bytes, INPUT_DIM + 1);
    set_e2m1_one(&mut weight_bytes, 2 * INPUT_DIM + 2);

    let input_bytes_dev = DeviceBuffer::from_host(&stream, &input_bytes)?;
    let input_scales_dev = DeviceBuffer::from_host(&stream, &one_scales(TOKEN_COUNT * INPUT_DIM))?;
    let input_global_scales_dev = DeviceBuffer::from_host(&stream, &[1.0_f32; TOKEN_COUNT])?;
    let weight_bytes_dev = DeviceBuffer::from_host(&stream, &weight_bytes)?;
    let weight_scales_dev = DeviceBuffer::from_host(&stream, &one_scales(VOCAB_SIZE * INPUT_DIM))?;
    let weight_global_scale_dev = DeviceBuffer::from_host(&stream, &[1.0_f32])?;
    let mut logits_dev = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * VOCAB_SIZE)?;

    module.logits(LmHeadArgs {
        stream: &stream,
        input: Nvfp4RowwiseDeviceTensor::new(
            &input_bytes_dev,
            &input_scales_dev,
            &input_global_scales_dev,
        ),
        weight: Nvfp4FourSixMmaWeightTensor::new(
            &weight_bytes_dev,
            &weight_scales_dev,
            &weight_global_scale_dev,
        ),
        logits: &mut logits_dev,
        token_count: TOKEN_COUNT as u32,
        input_dim: INPUT_DIM as u32,
        vocab_size: VOCAB_SIZE as u32,
    })?;

    let logits = logits_dev.to_host_vec(&stream)?;
    common::assert_close(logits[0], 1.0, TOLERANCE);
    common::assert_close(logits[1], 1.0, TOLERANCE);
    common::assert_close(logits[2], 0.0, TOLERANCE);
    common::assert_close(logits[VOCAB_SIZE], 0.0, TOLERANCE);
    common::assert_close(logits[VOCAB_SIZE + 1], 0.0, TOLERANCE);
    common::assert_close(logits[VOCAB_SIZE + 2], 1.0, TOLERANCE);
    Ok(())
}
