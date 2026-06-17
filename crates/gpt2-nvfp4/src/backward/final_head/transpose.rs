use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::{
    Nvfp4DecodeModule, Nvfp4DecodeTransposeArgs, Nvfp4DeviceTensor,
    Nvfp4RowwiseDecodeTransposeArgs, Nvfp4RowwiseDeviceTensor,
};
use rust_kernels_cuda::transpose::{TransposeF32Args, TransposeModule};

use crate::{GPT2_CONTEXT_LEN, GPT2_N_EMBD, GPT2_VOCAB_SIZE};

pub(super) fn transpose_dlogits(
    module: &TransposeModule,
    stream: &CudaStream,
    dlogits: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    module.transpose_f32(TransposeF32Args {
        stream,
        input: dlogits,
        output: out,
        rows: GPT2_CONTEXT_LEN as u32,
        cols: GPT2_VOCAB_SIZE as u32,
    })
}

pub(super) fn decode_lm_head_weight(
    module: &Nvfp4DecodeModule,
    stream: &CudaStream,
    weight: Nvfp4DeviceTensor<'_>,
    out: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    module.decode_transpose_f32(Nvfp4DecodeTransposeArgs {
        stream,
        input: weight,
        output: out,
        rows: GPT2_VOCAB_SIZE as u32,
        cols: GPT2_N_EMBD as u32,
    })
}

pub(super) fn decode_final_normalized(
    module: &Nvfp4DecodeModule,
    stream: &CudaStream,
    final_normalized: Nvfp4RowwiseDeviceTensor<'_>,
    out: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    module.decode_rowwise_transpose_f32(Nvfp4RowwiseDecodeTransposeArgs {
        stream,
        input: final_normalized,
        output: out,
        rows: GPT2_CONTEXT_LEN as u32,
        cols: GPT2_N_EMBD as u32,
    })
}
