use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::linear_backward::{
    LinearBackwardArgs, LinearBackwardInputTranspose, LinearBackwardModule,
    LinearBackwardMsEdenArgs, LinearBackwardMsEdenScratch, LinearBackwardWeightTranspose,
    MsEdenOperandScratchBuffer,
};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tc_matmul::nvfp4_tc_matmul_padded_k;

mod common;
#[path = "common/nvfp4.rs"]
mod nvfp4_common;

use nvfp4_common::{first_col_one_bytes, first_row_one_bytes};

const TOKEN_COUNT: usize = 64;
const INPUT_DIM: usize = 64;
const OUTPUT_DIM: usize = 64;
const E2M1_ONE_PAIR: u8 = 0x22;
const E4M3_ONE: u8 = 0x38;
const TOLERANCE: f32 = 1.0e-7;

struct LinearBackwardMsEdenScratchBuffers {
    e_h: MsEdenOperandScratchBuffer,
    weight_t_h: MsEdenOperandScratchBuffer,
    e_t_h: MsEdenOperandScratchBuffer,
    input_t_h: MsEdenOperandScratchBuffer,
}

impl LinearBackwardMsEdenScratchBuffers {
    fn new(
        stream: &CudaStream,
        token_count: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self, DriverError> {
        let output_k = nvfp4_tc_matmul_padded_k(output_dim as u32) as usize;
        let token_k = nvfp4_tc_matmul_padded_k(token_count as u32) as usize;

        Ok(Self {
            e_h: MsEdenOperandScratchBuffer::new(stream, token_count * output_k, token_count)?,
            weight_t_h: MsEdenOperandScratchBuffer::new(stream, input_dim * output_k, input_dim)?,
            e_t_h: MsEdenOperandScratchBuffer::new(stream, output_dim * token_k, output_dim)?,
            input_t_h: MsEdenOperandScratchBuffer::new(stream, input_dim * token_k, input_dim)?,
        })
    }

    fn as_args(&mut self) -> LinearBackwardMsEdenScratch<'_> {
        LinearBackwardMsEdenScratch {
            e_h: self.e_h.as_arg(),
            weight_t_h: self.weight_t_h.as_arg(),
            e_t_h: self.e_t_h.as_arg(),
            input_t_h: self.input_t_h.as_arg(),
        }
    }
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn linear_backward_computes_dinput_and_dweight_from_quartet_operands() -> Result<(), Box<dyn Error>>
{
    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = LinearBackwardModule::from_module(ptx)?;

    let e_h_bytes = first_col_one_bytes(TOKEN_COUNT, OUTPUT_DIM);
    let e_h_scales = vec![E4M3_ONE; TOKEN_COUNT * OUTPUT_DIM / 16];
    let e_h_global_scales = vec![1.0_f32; TOKEN_COUNT];

    let weight_t_h_bytes = first_col_one_bytes(INPUT_DIM, OUTPUT_DIM);
    let weight_t_h_scales = vec![E4M3_ONE; INPUT_DIM * OUTPUT_DIM / 16];

    let e_t_h_bytes = first_row_one_bytes(OUTPUT_DIM, TOKEN_COUNT);
    let e_t_h_scales = vec![E4M3_ONE; OUTPUT_DIM * TOKEN_COUNT / 16];
    let e_t_h_global_scales = vec![1.0_f32; OUTPUT_DIM];

    let input_t_h_bytes = vec![E2M1_ONE_PAIR; INPUT_DIM * TOKEN_COUNT / 2];
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
            common::assert_close(dweight[row * INPUT_DIM + col], expected, TOLERANCE);
        }
    }

    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn linear_backward_ms_eden_quantizes_before_gemms() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = LinearBackwardModule::from_module(ptx.clone())?;
    let quant_module = Nvfp4QuantModule::from_module(ptx)?;

    let e = patterned_matrix(TOKEN_COUNT, OUTPUT_DIM, 0.015625);
    let weight_t = patterned_matrix(INPUT_DIM, OUTPUT_DIM, 0.03125);
    let input_t = patterned_matrix(INPUT_DIM, TOKEN_COUNT, 0.03125);

    let e_dev = DeviceBuffer::from_host(&stream, &e)?;
    let weight_t_dev = DeviceBuffer::from_host(&stream, &weight_t)?;
    let input_t_dev = DeviceBuffer::from_host(&stream, &input_t)?;
    let mut scratch =
        LinearBackwardMsEdenScratchBuffers::new(&stream, TOKEN_COUNT, INPUT_DIM, OUTPUT_DIM)?;

    let mut dinput_dev = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_COUNT * INPUT_DIM)?;
    let mut dweight_dev = DeviceBuffer::<f32>::zeroed(&stream, OUTPUT_DIM * INPUT_DIM)?;
    let mut dbias_dev = DeviceBuffer::<f32>::zeroed(&stream, OUTPUT_DIM)?;

    module.backward_ms_eden(LinearBackwardMsEdenArgs {
        stream: &stream,
        quant_module: &quant_module,
        e: &e_dev,
        weight_t: LinearBackwardWeightTranspose::Fp32(&weight_t_dev),
        input_t: LinearBackwardInputTranspose::Fp32(&input_t_dev),
        scratch: scratch.as_args(),
        dinput: &mut dinput_dev,
        dweight: &mut dweight_dev,
        dbias: Some(&mut dbias_dev),
        token_count: TOKEN_COUNT as u32,
        input_dim: INPUT_DIM as u32,
        output_dim: OUTPUT_DIM as u32,
        sign_seed: 0x1234_5678,
        scale_seed: 0x9abc_def0,
        precomputed_e_amax_chunks: None,
    })?;

    let dinput = dinput_dev.to_host_vec(&stream)?;
    let dweight = dweight_dev.to_host_vec(&stream)?;
    let dbias = dbias_dev.to_host_vec(&stream)?;
    let e_quant = scratch.e_h.bytes.to_host_vec(&stream)?;
    let e_amax = scratch.e_h.chunk_amax.to_host_vec(&stream)?;
    let generated_scales = [
        scratch.e_h.global_scale.to_host_vec(&stream)?[0],
        scratch.weight_t_h.global_scale.to_host_vec(&stream)?[0],
        scratch.input_t_h.global_scale.to_host_vec(&stream)?[0],
    ];

    assert!(dinput.iter().all(|value| value.is_finite()));
    assert!(dweight.iter().all(|value| value.is_finite()));
    assert!(dinput.iter().any(|value| value.abs() > TOLERANCE));
    assert!(dweight.iter().any(|value| value.abs() > TOLERANCE));
    for col in 0..OUTPUT_DIM {
        let expected = (0..TOKEN_COUNT)
            .map(|row| e[row * OUTPUT_DIM + col])
            .sum::<f32>();
        common::assert_close(dbias[col], expected, TOLERANCE);
    }
    assert!(e_quant.iter().any(|byte| *byte != 0));
    assert!(e_amax.iter().all(|amax| amax.is_finite()));
    assert!(e_amax.iter().any(|amax| *amax > 0.0));
    assert!(
        generated_scales
            .iter()
            .all(|scale| *scale > 0.0 && scale.is_finite())
    );

    Ok(())
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
