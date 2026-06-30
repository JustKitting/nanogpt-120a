use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy, DriverError};

pub(super) fn zero<T: DeviceCopy>(
    stream: &CudaStream,
    len: usize,
) -> Result<DeviceBuffer<T>, DriverError> {
    DeviceBuffer::zeroed(stream, len)
}
