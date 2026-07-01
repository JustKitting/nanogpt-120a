use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::nvfp4::{
    Nvfp4DecodeModule, Nvfp4DecodeTransposeArgs, Nvfp4RowwiseDecodeTransposeArgs,
};
use rust_kernels_cuda::nvfp4_quant::{
    MsEdenDeviceScaleQuantArgs, MsEdenTransposeDeviceScaleQuantArgs, Nvfp4QuantArgs,
    Nvfp4QuantModule, Nvfp4TransposeMsEdenDeviceScaleQuantArgs,
};
use rust_kernels_cuda::transpose::{TransposeF32Args, TransposeModule};

mod common;
#[path = "ms_eden_transpose/support.rs"]
mod support;

use support::*;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn fp32_transpose_ms_eden_matches_materialized_transpose() -> Result<(), Box<dyn Error>> {
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

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn rowwise_nvfp4_transpose_ms_eden_matches_materialized_decode() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let decode = Nvfp4DecodeModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;

    let x = input_matrix();
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let mut source = RowwiseSourceScratch::new(&stream)?;
    let mut x_t_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS * COLS)?;
    let mut materialized = QuantScratch::new(&stream)?;
    let mut direct = QuantScratch::new(&stream)?;

    source.quantize(&stream, &quant, &x_dev, ROWS, COLS)?;

    let source_tensor = source.tensor();
    decode.decode_rowwise_transpose_f32(Nvfp4RowwiseDecodeTransposeArgs {
        stream: &stream,
        input: source_tensor,
        output: &mut x_t_dev,
        rows: ROWS as u32,
        cols: COLS as u32,
    })?;
    quant.fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale(
        materialized.quartet_args(&stream, &x_t_dev, COLS, ROWS, padded_rows()),
    )?;
    quant.rowwise_nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale(
        direct.rowwise_transpose_args(&stream, source_tensor, ROWS, COLS, padded_rows()),
    )?;

    direct.assert_quartet_eq(&stream, &materialized)?;
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn rowwise_nvfp4_transpose_no_chunk_no_pad_matches_materialized_decode()
-> Result<(), Box<dyn Error>> {
    const LOCAL_ROWS: usize = 32;
    const LOCAL_COLS: usize = 16;

    let (_, stream, ptx) = common::cuda_test_context()?;
    let decode = Nvfp4DecodeModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;

    let x = (0..LOCAL_ROWS * LOCAL_COLS)
        .map(|index| {
            let row = index / LOCAL_COLS;
            let col = index % LOCAL_COLS;
            (row as f32 - 12.0) * 0.015625 + (col as f32 - 5.0) * 0.03125
        })
        .collect::<Vec<_>>();
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let mut source = RowwiseSourceScratch::new_for_shape(&stream, LOCAL_ROWS, LOCAL_COLS)?;
    let mut x_t_dev = DeviceBuffer::<f32>::zeroed(&stream, x.len())?;
    let mut materialized = QuantScratch::new_exact(&stream, LOCAL_COLS, LOCAL_ROWS)?;
    let mut direct = QuantScratch::new_exact(&stream, LOCAL_COLS, LOCAL_ROWS)?;

    source.quantize(&stream, &quant, &x_dev, LOCAL_ROWS, LOCAL_COLS)?;

    let source_tensor = source.tensor();
    decode.decode_rowwise_transpose_f32(Nvfp4RowwiseDecodeTransposeArgs {
        stream: &stream,
        input: source_tensor,
        output: &mut x_t_dev,
        rows: LOCAL_ROWS as u32,
        cols: LOCAL_COLS as u32,
    })?;
    quant.fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
        materialized.quartet_args(&stream, &x_t_dev, LOCAL_COLS, LOCAL_ROWS, LOCAL_ROWS),
    )?;
    quant.rowwise_nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
        direct.rowwise_transpose_args(&stream, source_tensor, LOCAL_ROWS, LOCAL_COLS, LOCAL_ROWS),
    )?;

    direct.assert_no_chunk_quartet_eq(&stream, &materialized)?;
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_transpose_ms_eden_matches_materialized_decode() -> Result<(), Box<dyn Error>> {
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
