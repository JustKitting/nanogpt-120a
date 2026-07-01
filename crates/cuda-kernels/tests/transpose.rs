use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::transpose::{TransposeF32Args, TransposeModule};

mod common;

const ROWS: usize = 7;
const COLS: usize = 11;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn transpose_f32_writes_row_major_transpose() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(TransposeModule::from_module)?;

    let input = input_matrix();
    let input_dev = DeviceBuffer::from_host(&stream, &input)?;
    let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS * COLS)?;

    module.transpose_f32(TransposeF32Args {
        stream: &stream,
        input: &input_dev,
        output: &mut output_dev,
        rows: ROWS as u32,
        cols: COLS as u32,
    })?;

    let output = output_dev.to_host_vec(&stream)?;
    assert_eq!(output, expected_transpose(&input));
    Ok(())
}

fn input_matrix() -> Vec<f32> {
    (0..ROWS * COLS)
        .map(|index| {
            let row = index / COLS;
            let col = index % COLS;
            row as f32 * 100.0 + col as f32
        })
        .collect()
}

fn expected_transpose(input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0_f32; ROWS * COLS];
    for row in 0..ROWS {
        for col in 0..COLS {
            output[col * ROWS + row] = input[row * COLS + col];
        }
    }
    output
}
