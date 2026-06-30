use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy, DriverError};
use gpt2_nvfp4::GPT2_N_LAYER;

pub(super) fn zero<T: DeviceCopy>(
    stream: &CudaStream,
    len: usize,
) -> Result<DeviceBuffer<T>, DriverError> {
    DeviceBuffer::zeroed(stream, len)
}

pub(super) fn block_array<F, T>(f: F) -> Result<[T; GPT2_N_LAYER], DriverError>
where
    F: FnMut(usize) -> Result<T, DriverError>,
{
    let values = (0..GPT2_N_LAYER).map(f).collect::<Result<Vec<_>, _>>()?;
    match values.try_into() {
        Ok(array) => Ok(array),
        Err(_) => unreachable!("block array length must match GPT2_N_LAYER"),
    }
}
