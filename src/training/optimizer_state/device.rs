use cuda_core::{CudaStream, DeviceBuffer, DriverError, memory};
use gpt2_nvfp4::GPT2_N_LAYER;
use rust_kernels_cuda::nvfp4::{Nvfp4DecodeModule, Nvfp4DecodeTransposeArgs, Nvfp4DeviceTensor};

use crate::upload::UploadedNvfp4;

pub(super) fn decode_master(
    stream: &CudaStream,
    decode: &Nvfp4DecodeModule,
    tensor: &UploadedNvfp4,
) -> Result<DeviceBuffer<f32>, DriverError> {
    let mut master = DeviceBuffer::zeroed(stream, tensor.len)?;
    decode.decode_transpose_f32(Nvfp4DecodeTransposeArgs {
        stream,
        input: Nvfp4DeviceTensor {
            bytes: &tensor.bytes,
            scales: &tensor.scales,
            global_scale: &tensor.global_scale,
        },
        output: &mut master,
        rows: 1,
        cols: tensor.len as u32,
    })?;
    Ok(master)
}

pub(super) fn clone_device(
    stream: &CudaStream,
    buffer: &DeviceBuffer<f32>,
) -> Result<DeviceBuffer<f32>, DriverError> {
    let cloned = DeviceBuffer::zeroed(stream, buffer.len())?;
    stream.context().bind_to_thread()?;

    unsafe {
        memory::memcpy_dtod_async(
            cloned.cu_deviceptr(),
            buffer.cu_deviceptr(),
            buffer.num_bytes(),
            stream.cu_stream(),
        )?;
    }

    Ok(cloned)
}

pub(super) fn block_array<F, T>(mut f: F) -> Result<[T; GPT2_N_LAYER], DriverError>
where
    F: FnMut(usize) -> Result<T, DriverError>,
{
    let values = (0..GPT2_N_LAYER)
        .map(|i| f(i))
        .collect::<Result<Vec<_>, _>>()?;
    match values.try_into() {
        Ok(array) => Ok(array),
        Err(_) => unreachable!("block array length must match GPT2_N_LAYER"),
    }
}
