use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;
use rust_kernels_cuda::nvfp4_quant::{Nvfp4QuantModule, Nvfp4QuantRowwiseArgs, RowAmaxArgs};

use super::forward::NextLatForwardArgs;

pub(super) fn quantize_input(args: &mut NextLatForwardArgs<'_, '_>) -> Result<(), DriverError> {
    quantize_precomputed_amax(
        args.quant,
        args.stream,
        &args.buffers.normalized,
        &args.buffers.normalized_amax,
        RowwiseOut {
            bytes: &mut args.buffers.input_bytes,
            scales: &mut args.buffers.input_scales,
            global_scales: &mut args.buffers.input_globals,
        },
        args.row_count,
        gpt2_nvfp4::NEXTLAT_INPUT as u32,
    )
}

pub(super) fn quantize_activation(
    quant: &Nvfp4QuantModule,
    stream: &CudaStream,
    row_count: u32,
    input: &DeviceBuffer<f32>,
    amax: &mut DeviceBuffer<f32>,
    bytes: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<u8>,
    globals: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    quant.row_amax_f32(RowAmaxArgs {
        stream,
        x: input,
        out: amax,
        row_count,
        row_len: gpt2_nvfp4::NEXTLAT_HIDDEN as u32,
    })?;
    quantize_precomputed_amax(
        quant,
        stream,
        input,
        amax,
        RowwiseOut {
            bytes,
            scales,
            global_scales: globals,
        },
        row_count,
        gpt2_nvfp4::NEXTLAT_HIDDEN as u32,
    )
}

pub(super) fn rowwise<'a>(
    bytes: &'a DeviceBuffer<u8>,
    scales: &'a DeviceBuffer<u8>,
    global_scales: &'a DeviceBuffer<f32>,
) -> Nvfp4RowwiseDeviceTensor<'a> {
    Nvfp4RowwiseDeviceTensor {
        bytes,
        scales,
        global_scales,
    }
}

struct RowwiseOut<'a> {
    bytes: &'a mut DeviceBuffer<u8>,
    scales: &'a mut DeviceBuffer<u8>,
    global_scales: &'a mut DeviceBuffer<f32>,
}

fn quantize_precomputed_amax(
    quant: &Nvfp4QuantModule,
    stream: &CudaStream,
    input: &DeviceBuffer<f32>,
    amax: &DeviceBuffer<f32>,
    out: RowwiseOut<'_>,
    row_count: u32,
    row_len: u32,
) -> Result<(), DriverError> {
    quant.fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
        stream,
        x: input,
        amax,
        out_fp4: out.bytes,
        out_scales: out.scales,
        out_global_scale: out.global_scales,
        group_count: row_count * row_len / 16,
        row_len,
    })
}
