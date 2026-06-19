use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::nvfp4::{
    Nvfp4DecodeModule, Nvfp4DecodeTransposeArgs, Nvfp4DeviceTensor,
    Nvfp4RowwiseDecodeTransposeArgs, Nvfp4RowwiseDeviceTensor,
};
use rust_kernels_cuda::nvfp4_quant::{
    MsEdenDeviceScaleQuantArgs, MsEdenTransposeDeviceScaleQuantArgs, Nvfp4QuantArgs,
    Nvfp4QuantModule, Nvfp4QuantRowwiseArgs, Nvfp4TransposeMsEdenDeviceScaleQuantArgs,
    QuartetBackwardMsEdenDeviceScaleQuantArgs, RowAmaxArgs,
    RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs,
};
use rust_kernels_cuda::nvfp4_tc_matmul::{
    nvfp4_tc_matmul_bytes, nvfp4_tc_matmul_chunks, nvfp4_tc_matmul_padded_k, nvfp4_tc_matmul_scales,
};
use rust_kernels_cuda::transpose::{TransposeF32Args, TransposeModule};

mod common;

const ROWS: usize = 33;
const COLS: usize = 16;
const SIGN_SEED: u32 = 0x1c69_b3f5;
const SCALE_SEED: u32 = 0x4a7c_15d3;
const SCALE_OVERRIDE: f32 = 0.25;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn fp32_transpose_ms_eden_matches_materialized_transpose() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(common::ptx_path().as_str())?;
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

    assert_eq!(
        direct.bytes.to_host_vec(&stream)?,
        materialized.bytes.to_host_vec(&stream)?
    );
    assert_eq!(
        direct.scales.to_host_vec(&stream)?,
        materialized.scales.to_host_vec(&stream)?
    );
    assert_eq!(
        direct.global_scales.to_host_vec(&stream)?,
        materialized.global_scales.to_host_vec(&stream)?
    );
    assert_eq!(
        direct.chunk_amax.to_host_vec(&stream)?,
        materialized.chunk_amax.to_host_vec(&stream)?
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn rowwise_nvfp4_transpose_ms_eden_matches_materialized_decode() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(common::ptx_path().as_str())?;
    let decode = Nvfp4DecodeModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;

    let x = input_matrix();
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let mut source = RowwiseSourceScratch::new(&stream)?;
    let mut row_amax = DeviceBuffer::<f32>::zeroed(&stream, ROWS)?;
    let mut x_t_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS * COLS)?;
    let mut materialized = QuantScratch::new(&stream)?;
    let mut direct = QuantScratch::new(&stream)?;

    quant.row_amax_f32(RowAmaxArgs {
        stream: &stream,
        x: &x_dev,
        out: &mut row_amax,
        row_count: ROWS as u32,
        row_len: COLS as u32,
    })?;
    quant.fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
        stream: &stream,
        x: &x_dev,
        amax: &row_amax,
        out_fp4: &mut source.bytes,
        out_scales: &mut source.scales,
        out_global_scale: &mut source.global_scales,
        group_count: (ROWS * COLS / 16) as u32,
        row_len: COLS as u32,
    })?;

    let source_tensor = source.tensor();
    decode.decode_rowwise_transpose_f32(Nvfp4RowwiseDecodeTransposeArgs {
        stream: &stream,
        input: source_tensor,
        output: &mut x_t_dev,
        rows: ROWS as u32,
        cols: COLS as u32,
    })?;
    quant.fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale(
        QuartetBackwardMsEdenDeviceScaleQuantArgs {
            stream: &stream,
            x: &x_t_dev,
            out_fp4: &mut materialized.bytes,
            out_scales: &mut materialized.scales,
            out_global_scales: &mut materialized.global_scales,
            out_chunk_amax: &mut materialized.chunk_amax,
            out_global_scale: &mut materialized.global_scale,
            row_count: COLS as u32,
            src_row_len: ROWS as u32,
            dst_row_len: padded_rows() as u32,
            sign_seed: SIGN_SEED,
            scale_seed: SCALE_SEED,
        },
    )?;
    quant.rowwise_nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale(
        RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs {
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

    assert_eq!(
        direct.bytes.to_host_vec(&stream)?,
        materialized.bytes.to_host_vec(&stream)?
    );
    assert_eq!(
        direct.scales.to_host_vec(&stream)?,
        materialized.scales.to_host_vec(&stream)?
    );
    assert_eq!(
        direct.global_scales.to_host_vec(&stream)?,
        materialized.global_scales.to_host_vec(&stream)?
    );
    assert_eq!(
        direct.chunk_amax.to_host_vec(&stream)?,
        materialized.chunk_amax.to_host_vec(&stream)?
    );
    assert_eq!(
        direct.global_scale.to_host_vec(&stream)?,
        materialized.global_scale.to_host_vec(&stream)?
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_transpose_ms_eden_matches_materialized_decode() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(common::ptx_path().as_str())?;
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
        QuartetBackwardMsEdenDeviceScaleQuantArgs {
            stream: &stream,
            x: &x_t_dev,
            out_fp4: &mut materialized.bytes,
            out_scales: &mut materialized.scales,
            out_global_scales: &mut materialized.global_scales,
            out_chunk_amax: &mut materialized.chunk_amax,
            out_global_scale: &mut materialized.global_scale,
            row_count: COLS as u32,
            src_row_len: ROWS as u32,
            dst_row_len: padded_rows() as u32,
            sign_seed: SIGN_SEED,
            scale_seed: SCALE_SEED,
        },
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

    assert_eq!(
        direct.bytes.to_host_vec(&stream)?,
        materialized.bytes.to_host_vec(&stream)?
    );
    assert_eq!(
        direct.scales.to_host_vec(&stream)?,
        materialized.scales.to_host_vec(&stream)?
    );
    assert_eq!(
        direct.global_scales.to_host_vec(&stream)?,
        materialized.global_scales.to_host_vec(&stream)?
    );
    assert_eq!(
        direct.chunk_amax.to_host_vec(&stream)?,
        materialized.chunk_amax.to_host_vec(&stream)?
    );
    assert_eq!(
        direct.global_scale.to_host_vec(&stream)?,
        materialized.global_scale.to_host_vec(&stream)?
    );
    Ok(())
}

struct QuantScratch {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scales: DeviceBuffer<f32>,
    chunk_amax: DeviceBuffer<f32>,
    global_scale: DeviceBuffer<f32>,
}

impl QuantScratch {
    fn new(stream: &cuda_core::CudaStream) -> Result<Self, cuda_core::DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_bytes(COLS as u32, ROWS as u32))?,
            scales: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_scales(COLS as u32, ROWS as u32))?,
            global_scales: DeviceBuffer::zeroed(stream, COLS)?,
            chunk_amax: DeviceBuffer::zeroed(
                stream,
                nvfp4_tc_matmul_chunks(COLS as u32, ROWS as u32),
            )?,
            global_scale: DeviceBuffer::zeroed(stream, 1)?,
        })
    }
}

struct SourceScratch {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scale: DeviceBuffer<f32>,
}

impl SourceScratch {
    fn new(stream: &cuda_core::CudaStream) -> Result<Self, cuda_core::DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, ROWS * COLS / 2)?,
            scales: DeviceBuffer::zeroed(stream, ROWS * COLS / 16)?,
            global_scale: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    fn tensor(&self) -> Nvfp4DeviceTensor<'_> {
        Nvfp4DeviceTensor {
            bytes: &self.bytes,
            scales: &self.scales,
            global_scale: &self.global_scale,
        }
    }
}

struct RowwiseSourceScratch {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scales: DeviceBuffer<f32>,
}

impl RowwiseSourceScratch {
    fn new(stream: &cuda_core::CudaStream) -> Result<Self, cuda_core::DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, ROWS * COLS / 2)?,
            scales: DeviceBuffer::zeroed(stream, ROWS * COLS / 16)?,
            global_scales: DeviceBuffer::zeroed(stream, ROWS)?,
        })
    }

    fn tensor(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor {
            bytes: &self.bytes,
            scales: &self.scales,
            global_scales: &self.global_scales,
        }
    }
}

fn padded_rows() -> usize {
    nvfp4_tc_matmul_padded_k(ROWS as u32) as usize
}

fn input_matrix() -> Vec<f32> {
    (0..ROWS * COLS)
        .map(|index| {
            let row = index / COLS;
            let col = index % COLS;
            (row as f32 - 9.0) * 0.03125 + (col as f32 - 4.0) * 0.0078125
        })
        .collect()
}

fn cpu_amax(x: &[f32]) -> f32 {
    x.iter().fold(0.0, |max, value| max.max(value.abs()))
}
