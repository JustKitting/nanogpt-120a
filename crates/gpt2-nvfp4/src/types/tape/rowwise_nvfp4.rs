use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use super::device_copy::copy_device;
use super::types::RowwiseNvfp4Tape;

impl<'a> RowwiseNvfp4Tape<'a> {
    pub(crate) fn saved(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor::new(&*self.bytes, &*self.scales, &*self.global_scales)
    }

    pub(crate) fn reborrow(&mut self) -> RowwiseNvfp4Tape<'_> {
        RowwiseNvfp4Tape {
            bytes: &mut *self.bytes,
            scales: &mut *self.scales,
            global_scales: &mut *self.global_scales,
        }
    }

    pub(crate) fn save(
        &mut self,
        stream: &CudaStream,
        src: Nvfp4RowwiseDeviceTensor<'_>,
    ) -> Result<(), DriverError> {
        copy_device(stream, src.bytes, self.bytes)?;
        copy_device(stream, src.scales, self.scales)?;
        copy_device(stream, src.global_scales, self.global_scales)
    }
}
