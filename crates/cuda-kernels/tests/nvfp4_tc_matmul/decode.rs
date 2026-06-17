use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::{
    Nvfp4DecodeModule, Nvfp4RowwiseDecodeTransposeArgs, Nvfp4RowwiseDeviceTensor,
};

use super::scratch::{M, N, ScratchBuffers, padded_k};

pub fn decoded_dot(
    decode: &Nvfp4DecodeModule,
    stream: &CudaStream,
    scratch: &ScratchBuffers,
) -> Result<f32, DriverError> {
    let mut a_t = DeviceBuffer::<f32>::zeroed(stream, M * padded_k())?;
    let mut b_t_t = DeviceBuffer::<f32>::zeroed(stream, N * padded_k())?;
    decode.decode_rowwise_transpose_f32(decode_args(stream, scratch.a_tensor(), &mut a_t, M))?;
    decode.decode_rowwise_transpose_f32(decode_args(stream, scratch.b_tensor(), &mut b_t_t, N))?;
    let a = a_t.to_host_vec(stream)?;
    let b = b_t_t.to_host_vec(stream)?;
    Ok((0..padded_k()).map(|i| a[i * M] * b[i * N]).sum())
}

fn decode_args<'a>(
    stream: &'a CudaStream,
    input: Nvfp4RowwiseDeviceTensor<'a>,
    output: &'a mut DeviceBuffer<f32>,
    rows: usize,
) -> Nvfp4RowwiseDecodeTransposeArgs<'a, 'a> {
    Nvfp4RowwiseDecodeTransposeArgs {
        stream,
        input,
        output,
        rows: rows as u32,
        cols: padded_k() as u32,
    }
}
