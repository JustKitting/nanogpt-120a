use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::{
    Nvfp4DecodeModule, Nvfp4DecodeTransposeArgs, Nvfp4DeviceTensor,
    Nvfp4RowwiseDecodeTransposeArgs, Nvfp4RowwiseDeviceTensor,
};
use rust_kernels_cuda::transpose::{TransposeF32Args, TransposeModule};

pub(super) fn decode_weight_t(
    module: &Nvfp4DecodeModule,
    stream: &CudaStream,
    weight: Nvfp4FourSixMmaWeightTensor<'_>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
) -> Result<(), DriverError> {
    module.decode_transpose_f32(Nvfp4DecodeTransposeArgs {
        stream,
        input: Nvfp4DeviceTensor {
            bytes: weight.bytes,
            scales: weight.scales,
            global_scale: weight.global_scale,
        },
        output,
        rows: rows as u32,
        cols: cols as u32,
    })
}

pub(super) fn decode_rowwise_t(
    module: &Nvfp4DecodeModule,
    stream: &CudaStream,
    input: Nvfp4RowwiseDeviceTensor<'_>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
) -> Result<(), DriverError> {
    module.decode_rowwise_transpose_f32(Nvfp4RowwiseDecodeTransposeArgs {
        stream,
        input,
        output,
        rows: rows as u32,
        cols: cols as u32,
    })
}

pub(super) fn transpose_f32(
    module: &TransposeModule,
    stream: &CudaStream,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
) -> Result<(), DriverError> {
    module.transpose_f32(TransposeF32Args {
        stream,
        input,
        output,
        rows: rows as u32,
        cols: cols as u32,
    })
}
