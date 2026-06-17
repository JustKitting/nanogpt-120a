use cuda_core::{CudaStream, DeviceBuffer, DriverError, memory};

pub(crate) fn copy_device<T>(
    stream: &CudaStream,
    src: &DeviceBuffer<T>,
    dst: &mut DeviceBuffer<T>,
) -> Result<(), DriverError> {
    assert_eq!(src.len(), dst.len());
    stream.context().bind_to_thread()?;

    unsafe {
        memory::memcpy_dtod_async(
            dst.cu_deviceptr(),
            src.cu_deviceptr(),
            src.num_bytes(),
            stream.cu_stream(),
        )
    }
}
