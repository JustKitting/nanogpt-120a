use cuda_core::DeviceBuffer;
use rust_kernels_cuda::nvfp4_quant::{
    MsEdenDeviceScaleQuantArgs, MsEdenTransposeDeviceScaleQuantArgs, Nvfp4QuantModule,
};
use rust_kernels_cuda::transpose::{TransposeF32Args, TransposeModule};

use super::TestResult;
use super::common;
use super::support::*;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn fp32_transpose_ms_eden_matches_materialized_transpose() -> TestResult {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let transpose = TransposeModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;

    let x = input_matrix();
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let mut x_t_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS * COLS)?;
    let global_scale = DeviceBuffer::from_host(&stream, &[0.75_f32])?;
    let mut materialized = QuantScratch::new(&stream)?;
    let mut direct = QuantScratch::new(&stream)?;

    transpose.transpose_f32(TransposeF32Args {
        stream: &stream,
        input: &x_dev,
        output: &mut x_t_dev,
        rows: ROWS as u32,
        cols: COLS as u32,
    })?;

    quant.fp32_to_nvfp4_ms_eden_device_scale(MsEdenDeviceScaleQuantArgs {
        stream: &stream,
        x: &x_t_dev,
        out_fp4: &mut materialized.bytes,
        out_scales: &mut materialized.scales,
        out_global_scales: &mut materialized.global_scales,
        out_chunk_amax: &mut materialized.chunk_amax,
        global_scale: &global_scale,
        row_count: COLS as u32,
        src_row_len: ROWS as u32,
        dst_row_len: padded_rows() as u32,
        scale_override: SCALE_OVERRIDE,
        sign_seed: SIGN_SEED,
        scale_seed: SCALE_SEED,
    })?;

    quant.fp32_transpose_to_nvfp4_ms_eden_device_scale(MsEdenTransposeDeviceScaleQuantArgs {
        stream: &stream,
        x: &x_dev,
        out_fp4: &mut direct.bytes,
        out_scales: &mut direct.scales,
        out_global_scales: &mut direct.global_scales,
        out_chunk_amax: &mut direct.chunk_amax,
        global_scale: &global_scale,
        source_rows: ROWS as u32,
        source_cols: COLS as u32,
        dst_row_len: padded_rows() as u32,
        scale_override: SCALE_OVERRIDE,
        sign_seed: SIGN_SEED,
        scale_seed: SCALE_SEED,
    })?;

    direct.assert_ms_eden_eq(&stream, &materialized)?;
    Ok(())
}
