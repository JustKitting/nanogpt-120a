use cuda_core::DeviceBuffer;
use rust_kernels_cuda::nvfp4::{Nvfp4DecodeModule, Nvfp4DecodeTransposeArgs};
use rust_kernels_cuda::nvfp4_quant::{
    Nvfp4QuantArgs, Nvfp4QuantModule, Nvfp4TransposeMsEdenDeviceScaleQuantArgs,
};

use super::TestResult;
use super::common;
use super::support::*;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_transpose_ms_eden_matches_materialized_decode() -> TestResult {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let decode = Nvfp4DecodeModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;

    let x = input_matrix();
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let mut source = SourceScratch::new(&stream)?;
    let amax = DeviceBuffer::from_host(&stream, &[cpu_amax(&x)])?;
    let mut x_t_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS * COLS)?;
    let mut materialized = QuantScratch::new(&stream)?;
    let mut direct = QuantScratch::new(&stream)?;

    quant.fp32_to_nvfp4_four_six(Nvfp4QuantArgs {
        stream: &stream,
        x: &x_dev,
        amax: &amax,
        out_fp4: &mut source.bytes,
        out_scales: &mut source.scales,
        out_global_scale: &mut source.global_scale,
        group_count: (ROWS * COLS / 16) as u32,
    })?;

    let source_tensor = source.tensor();
    decode.decode_transpose_f32(Nvfp4DecodeTransposeArgs {
        stream: &stream,
        input: source_tensor,
        output: &mut x_t_dev,
        rows: ROWS as u32,
        cols: COLS as u32,
    })?;
    quant.fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale(
        materialized.quartet_args(&stream, &x_t_dev, COLS, ROWS, padded_rows()),
    )?;
    quant.nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale(
        Nvfp4TransposeMsEdenDeviceScaleQuantArgs {
            stream: &stream,
            input: source_tensor,
            out_fp4: &mut direct.bytes,
            out_scales: &mut direct.scales,
            out_global_scales: &mut direct.global_scales,
            out_chunk_amax: &mut direct.chunk_amax,
            out_global_scale: &mut direct.global_scale,
            source_rows: ROWS as u32,
            source_cols: COLS as u32,
            dst_row_len: padded_rows() as u32,
            sign_seed: SIGN_SEED,
            scale_seed: SCALE_SEED,
        },
    )?;

    direct.assert_quartet_eq(&stream, &materialized)?;
    Ok(())
}
