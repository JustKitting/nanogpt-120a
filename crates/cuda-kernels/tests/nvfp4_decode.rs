use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::nvfp4::{
    Nvfp4DecodeModule, Nvfp4DecodeTransposeArgs, Nvfp4DeviceTensor,
    Nvfp4RowwiseDecodeTransposeArgs, Nvfp4RowwiseDeviceTensor,
};

mod common;

use common::nvfp4::{one_pair_bytes, one_scales};

const ROWS: usize = 2;
const COLS: usize = 16;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_decode_transpose_writes_fp32_transpose() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(Nvfp4DecodeModule::from_module)?;

    let bytes = DeviceBuffer::from_host(&stream, &one_pair_bytes(ROWS * COLS))?;
    let scales = DeviceBuffer::from_host(&stream, &one_scales(ROWS * COLS))?;
    let scalar_global_scale = DeviceBuffer::from_host(&stream, &[3.0_f32])?;
    let mut scalar_out = DeviceBuffer::<f32>::zeroed(&stream, ROWS * COLS)?;
    let globals = DeviceBuffer::from_host(&stream, &[1.0_f32, 2.0])?;
    let mut rowwise_out = DeviceBuffer::<f32>::zeroed(&stream, ROWS * COLS)?;

    module.decode_transpose_f32(Nvfp4DecodeTransposeArgs {
        stream: &stream,
        input: Nvfp4DeviceTensor::new(&bytes, &scales, &scalar_global_scale),
        output: &mut scalar_out,
        rows: ROWS as u32,
        cols: COLS as u32,
    })?;

    module.decode_rowwise_transpose_f32(Nvfp4RowwiseDecodeTransposeArgs {
        stream: &stream,
        input: Nvfp4RowwiseDeviceTensor::new(&bytes, &scales, &globals),
        output: &mut rowwise_out,
        rows: ROWS as u32,
        cols: COLS as u32,
    })?;

    assert!(scalar_out.to_host_vec(&stream)?.iter().all(|v| *v == 3.0));
    assert_eq!(rowwise_out.to_host_vec(&stream)?, expected_rowwise());
    Ok(())
}

fn expected_rowwise() -> Vec<f32> {
    let mut output = vec![0.0_f32; ROWS * COLS];
    for col in 0..COLS {
        output[col * ROWS] = 1.0;
        output[col * ROWS + 1] = 2.0;
    }
    output
}
