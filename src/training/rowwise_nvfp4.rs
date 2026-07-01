use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{RowwiseNvfp4Scratch, RowwiseNvfp4Tape};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use super::device_buffer::zero;

pub(crate) struct RowwiseNvfp4Buffers {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scales: DeviceBuffer<f32>,
}

impl RowwiseNvfp4Buffers {
    pub(crate) fn new(
        stream: &CudaStream,
        elements: usize,
        rows: usize,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: zero(stream, elements / 2)?,
            scales: zero(stream, elements / 16)?,
            global_scales: zero(stream, rows)?,
        })
    }

    pub(crate) fn scratch(&mut self) -> RowwiseNvfp4Scratch<'_> {
        RowwiseNvfp4Scratch {
            bytes: &mut self.bytes,
            scales: &mut self.scales,
            global_scales: &mut self.global_scales,
        }
    }

    pub(crate) fn tape(&mut self) -> RowwiseNvfp4Tape<'_> {
        RowwiseNvfp4Tape {
            bytes: &mut self.bytes,
            scales: &mut self.scales,
            global_scales: &mut self.global_scales,
        }
    }

    pub(crate) fn rowwise(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor::new(&self.bytes, &self.scales, &self.global_scales)
    }

    pub(crate) fn saved(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        self.rowwise()
    }
}
