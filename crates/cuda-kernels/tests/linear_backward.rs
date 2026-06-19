use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::linear_backward::{
    LinearBackwardArgs, LinearBackwardInputTranspose, LinearBackwardModule,
    LinearBackwardMsEdenArgs, LinearBackwardMsEdenScratch, LinearBackwardWeightTranspose,
    MsEdenOperandScratch,
};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

mod common;

const TOKEN_COUNT: usize = 64;
const INPUT_DIM: usize = 8;
const OUTPUT_DIM: usize = 64;
const E4M3_ONE: u8 = 0x38;
const TOLERANCE: f32 = 1.0e-7;

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
    let weight_t_h_global_scale_dev = DeviceBuffer::from_host(&stream, &[1.0_f32])?;
    let e_t_h_bytes_dev = DeviceBuffer::from_host(&stream, &e_t_h_bytes)?;
    let e_t_h_scales_dev = DeviceBuffer::from_host(&stream, &e_t_h_scales)?;
    let e_t_h_global_scales_dev = DeviceBuffer::from_host(&stream, &e_t_h_global_scales)?;
    let input_t_h_bytes_dev = DeviceBuffer::from_host(&stream, &input_t_h_bytes)?;
    let input_t_h_scales_dev = DeviceBuffer::from_host(&stream, &input_t_h_scales)?;
    let input_t_h_global_scale_dev = DeviceBuffer::from_host(&stream, &[1.0_f32])?;
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
            global_scale: &weight_t_h_global_scale_dev,
        },
        e_t_h: Nvfp4RowwiseDeviceTensor {
            bytes: &e_t_h_bytes_dev,
            scales: &e_t_h_scales_dev,
            global_scales: &e_t_h_global_scales_dev,
        },
        input_t_h: Nvfp4FourSixMmaWeightTensor {
            bytes: &input_t_h_bytes_dev,
            scales: &input_t_h_scales_dev,
            global_scale: &input_t_h_global_scale_dev,
        },
        dinput: &mut dinput_dev,
        dweight: &mut dweight_dev,
        dbias: None,
        token_count: TOKEN_COUNT as u32,
        input_dim: INPUT_DIM as u32,
        output_dim: OUTPUT_DIM as u32,
    })?;

    let dinput = dinput_dev.to_host_vec(&stream)?;
    let dweight = dweight_dev.to_host_vec(&stream)?;

    assert!(dinput.iter().all(|value| (*value - 1.0).abs() <= TOLERANCE));

    for row in 0..OUTPUT_DIM {
        for col in 0..INPUT_DIM {
            let expected = if row == 0 { TOKEN_COUNT as f32 } else { 0.0 };
            assert_close(dweight[row * INPUT_DIM + col], expected);
        }
    }

    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn linear_backward_ms_eden_quantizes_before_gemms() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(common::ptx_path().as_str())?;
    let module = LinearBackwardModule::from_module(ptx.clone())?;
    let quant_module = Nvfp4QuantModule::from_module(ptx)?;

    let e = patterned_matrix(TOKEN_COUNT, OUTPUT_DIM, 0.015625);
    let weight_t = patterned_matrix(INPUT_DIM, OUTPUT_DIM, 0.03125);
    let input_t = patterned_matrix(INPUT_DIM, TOKEN_COUNT, 0.03125);

    let e_dev = DeviceBuffer::from_host(&stream, &e)?;
    let weight_t_dev = DeviceBuffer::from_host(&stream, &weight_t)?;
    let input_t_dev = DeviceBuffer::from_host(&stream, &input_t)?;

    let mut e_bytes = DeviceBuffer::<u8>::zeroed(&stream, TOKEN_COUNT * OUTPUT_DIM / 2)?;
    let mut e_scales = DeviceBuffer::<u8>::zeroed(&stream, TOKEN_COUNT * OUTPUT_DIM / 16)?;
    let mut e_global_scales = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT)?;
    let mut e_chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * OUTPUT_DIM / 32)?;
    let mut e_global_scale = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    let mut weight_t_bytes = DeviceBuffer::<u8>::zeroed(&stream, INPUT_DIM * OUTPUT_DIM / 2)?;
    let mut weight_t_scales = DeviceBuffer::<u8>::zeroed(&stream, INPUT_DIM * OUTPUT_DIM / 16)?;
    let mut weight_t_global_scales = DeviceBuffer::<f32>::zeroed(&stream, INPUT_DIM)?;
    let mut weight_t_chunk_amax =
        DeviceBuffer::<f32>::zeroed(&stream, INPUT_DIM * OUTPUT_DIM / 32)?;
    let mut weight_t_global_scale = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    let mut e_t_bytes = DeviceBuffer::<u8>::zeroed(&stream, OUTPUT_DIM * TOKEN_COUNT / 2)?;
    let mut e_t_scales = DeviceBuffer::<u8>::zeroed(&stream, OUTPUT_DIM * TOKEN_COUNT / 16)?;
    let mut e_t_global_scales = DeviceBuffer::<f32>::zeroed(&stream, OUTPUT_DIM)?;
    let mut e_t_chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, OUTPUT_DIM * TOKEN_COUNT / 32)?;
    let mut e_t_global_scale = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    let mut input_t_bytes = DeviceBuffer::<u8>::zeroed(&stream, INPUT_DIM * TOKEN_COUNT / 2)?;
    let mut input_t_scales = DeviceBuffer::<u8>::zeroed(&stream, INPUT_DIM * TOKEN_COUNT / 16)?;
    let mut input_t_global_scales = DeviceBuffer::<f32>::zeroed(&stream, INPUT_DIM)?;
    let mut input_t_chunk_amax =
        DeviceBuffer::<f32>::zeroed(&stream, INPUT_DIM * TOKEN_COUNT / 32)?;
    let mut input_t_global_scale = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    let mut dinput_dev = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * INPUT_DIM)?;
    let mut dweight_dev = DeviceBuffer::<f32>::zeroed(&stream, OUTPUT_DIM * INPUT_DIM)?;
    let mut dbias_dev = DeviceBuffer::<f32>::zeroed(&stream, OUTPUT_DIM)?;

    module.backward_ms_eden(LinearBackwardMsEdenArgs {
        stream: &stream,
        quant_module: &quant_module,
        e: &e_dev,
        weight_t: LinearBackwardWeightTranspose::Fp32(&weight_t_dev),
        input_t: LinearBackwardInputTranspose::Fp32(&input_t_dev),
        scratch: LinearBackwardMsEdenScratch {
            e_h: MsEdenOperandScratch {
                bytes: &mut e_bytes,
                scales: &mut e_scales,
                global_scales: &mut e_global_scales,
                chunk_amax: &mut e_chunk_amax,
                global_scale: &mut e_global_scale,
            },
            weight_t_h: MsEdenOperandScratch {
                bytes: &mut weight_t_bytes,
                scales: &mut weight_t_scales,
                global_scales: &mut weight_t_global_scales,
                chunk_amax: &mut weight_t_chunk_amax,
                global_scale: &mut weight_t_global_scale,
            },
            e_t_h: MsEdenOperandScratch {
                bytes: &mut e_t_bytes,
                scales: &mut e_t_scales,
                global_scales: &mut e_t_global_scales,
                chunk_amax: &mut e_t_chunk_amax,
                global_scale: &mut e_t_global_scale,
            },
            input_t_h: MsEdenOperandScratch {
                bytes: &mut input_t_bytes,
                scales: &mut input_t_scales,
                global_scales: &mut input_t_global_scales,
                chunk_amax: &mut input_t_chunk_amax,
                global_scale: &mut input_t_global_scale,
            },
        },
        dinput: &mut dinput_dev,
        dweight: &mut dweight_dev,
        dbias: Some(&mut dbias_dev),
        token_count: TOKEN_COUNT as u32,
        input_dim: INPUT_DIM as u32,
        output_dim: OUTPUT_DIM as u32,
        sign_seed: 0x1234_5678,
        scale_seed: 0x9abc_def0,
    })?;

    let dinput = dinput_dev.to_host_vec(&stream)?;
    let dweight = dweight_dev.to_host_vec(&stream)?;
    let dbias = dbias_dev.to_host_vec(&stream)?;
    let e_quant = e_bytes.to_host_vec(&stream)?;
    let e_amax = e_chunk_amax.to_host_vec(&stream)?;
    let generated_scales = [
        e_global_scale.to_host_vec(&stream)?[0],
        weight_t_global_scale.to_host_vec(&stream)?[0],
        input_t_global_scale.to_host_vec(&stream)?[0],
    ];

    assert!(dinput.iter().all(|value| value.is_finite()));
    assert!(dweight.iter().all(|value| value.is_finite()));
    assert!(dinput.iter().any(|value| value.abs() > TOLERANCE));
    assert!(dweight.iter().any(|value| value.abs() > TOLERANCE));
    for col in 0..OUTPUT_DIM {
        let expected = (0..TOKEN_COUNT)
            .map(|row| e[row * OUTPUT_DIM + col])
            .sum::<f32>();
        assert_close(dbias[col], expected);
    }
    assert!(e_quant.iter().any(|byte| *byte != 0));
    assert!(e_amax.iter().all(|amax| *amax > 0.0 && amax.is_finite()));
    assert!(
        generated_scales
            .iter()
            .all(|scale| *scale > 0.0 && scale.is_finite())
    );

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
        error <= TOLERANCE,
        "actual={actual:.8e} expected={expected:.8e} error={error:.8e}"
    );
}

fn patterned_matrix(rows: usize, cols: usize, scale: f32) -> Vec<f32> {
    (0..rows * cols)
        .map(|index| {
            let row = index / cols;
            let col = index % cols;
            ((row as f32 * 0.5) + (col as f32 - 31.5)) * scale
        })
        .collect()
}
