use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::linear_backward::{LinearBackwardArgs, LinearBackwardModule};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

mod common;

const TOKEN_COUNT: usize = 64;
const INPUT_DIM: usize = 8;
const OUTPUT_DIM: usize = 64;
const E4M3_ONE: u8 = 0x38;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn linear_backward_computes_dinput_and_dweight_from_quartet_operands() -> Result<(), Box<dyn Error>>
{
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        LinearBackwardModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;

    let mut e_h_bytes = vec![0_u8; TOKEN_COUNT * OUTPUT_DIM / 2];
    for row in 0..TOKEN_COUNT {
        set_e2m1_one(&mut e_h_bytes, row * OUTPUT_DIM);
    }
    let e_h_scales = vec![E4M3_ONE; TOKEN_COUNT * OUTPUT_DIM / 16];
    let e_h_global_scales = vec![1.0_f32; TOKEN_COUNT];

    let mut weight_t_h_bytes = vec![0_u8; INPUT_DIM * OUTPUT_DIM / 2];
    for row in 0..INPUT_DIM {
        set_e2m1_one(&mut weight_t_h_bytes, row * OUTPUT_DIM);
    }
    let weight_t_h_scales = vec![E4M3_ONE; INPUT_DIM * OUTPUT_DIM / 16];

    let mut e_t_h_bytes = vec![0_u8; OUTPUT_DIM * TOKEN_COUNT / 2];
    for col in 0..TOKEN_COUNT {
        set_e2m1_one(&mut e_t_h_bytes, col);
    }
    let e_t_h_scales = vec![E4M3_ONE; OUTPUT_DIM * TOKEN_COUNT / 16];
    let e_t_h_global_scales = vec![1.0_f32; OUTPUT_DIM];

    let mut input_t_h_bytes = vec![0_u8; INPUT_DIM * TOKEN_COUNT / 2];
    for row in 0..INPUT_DIM {
        for col in 0..TOKEN_COUNT {
            set_e2m1_one(&mut input_t_h_bytes, row * TOKEN_COUNT + col);
        }
    }
    let input_t_h_scales = vec![E4M3_ONE; INPUT_DIM * TOKEN_COUNT / 16];

    let e_h_bytes_dev = DeviceBuffer::from_host(&stream, &e_h_bytes)?;
    let e_h_scales_dev = DeviceBuffer::from_host(&stream, &e_h_scales)?;
    let e_h_global_scales_dev = DeviceBuffer::from_host(&stream, &e_h_global_scales)?;
    let weight_t_h_bytes_dev = DeviceBuffer::from_host(&stream, &weight_t_h_bytes)?;
    let weight_t_h_scales_dev = DeviceBuffer::from_host(&stream, &weight_t_h_scales)?;
    let e_t_h_bytes_dev = DeviceBuffer::from_host(&stream, &e_t_h_bytes)?;
    let e_t_h_scales_dev = DeviceBuffer::from_host(&stream, &e_t_h_scales)?;
    let e_t_h_global_scales_dev = DeviceBuffer::from_host(&stream, &e_t_h_global_scales)?;
    let input_t_h_bytes_dev = DeviceBuffer::from_host(&stream, &input_t_h_bytes)?;
    let input_t_h_scales_dev = DeviceBuffer::from_host(&stream, &input_t_h_scales)?;
    let mut dinput_dev = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * INPUT_DIM)?;
    let mut dweight_dev = DeviceBuffer::<f32>::zeroed(&stream, OUTPUT_DIM * INPUT_DIM)?;

    module.backward(LinearBackwardArgs {
        stream: &stream,
        e_h: Nvfp4RowwiseDeviceTensor {
            bytes: &e_h_bytes_dev,
            scales: &e_h_scales_dev,
            global_scales: &e_h_global_scales_dev,
        },
        weight_t_h: Nvfp4FourSixMmaWeightTensor {
            bytes: &weight_t_h_bytes_dev,
            scales: &weight_t_h_scales_dev,
            global_scale: 1.0,
        },
        e_t_h: Nvfp4RowwiseDeviceTensor {
            bytes: &e_t_h_bytes_dev,
            scales: &e_t_h_scales_dev,
            global_scales: &e_t_h_global_scales_dev,
        },
        input_t_h: Nvfp4FourSixMmaWeightTensor {
            bytes: &input_t_h_bytes_dev,
            scales: &input_t_h_scales_dev,
            global_scale: 1.0,
        },
        dinput: &mut dinput_dev,
        dweight: &mut dweight_dev,
        token_count: TOKEN_COUNT as u32,
        input_dim: INPUT_DIM as u32,
        output_dim: OUTPUT_DIM as u32,
    })?;

    let dinput = dinput_dev.to_host_vec(&stream)?;
    let dweight = dweight_dev.to_host_vec(&stream)?;

    assert!(dinput.iter().all(|value| (*value - 1.0).abs() <= 1.0e-5));

    for row in 0..OUTPUT_DIM {
        for col in 0..INPUT_DIM {
            let expected = if row == 0 { TOKEN_COUNT as f32 } else { 0.0 };
            assert_close(dweight[row * INPUT_DIM + col], expected);
        }
    }

    Ok(())
}

fn set_e2m1_one(bytes: &mut [u8], element: usize) {
    let byte = &mut bytes[element / 2];
    if element & 1 == 0 {
        *byte = (*byte & 0xf0) | 0x2;
    } else {
        *byte = (*byte & 0x0f) | 0x20;
    }
}

fn assert_close(actual: f32, expected: f32) {
    let error = (actual - expected).abs();
    assert!(
        error <= 1.0e-5,
        "actual={actual:.8e} expected={expected:.8e} error={error:.8e}"
    );
}
