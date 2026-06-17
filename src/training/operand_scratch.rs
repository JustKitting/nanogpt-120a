use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::linear_backward::MsEdenOperandScratch;

pub(super) struct OperandScratch {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scales: DeviceBuffer<f32>,
    chunk_amax: DeviceBuffer<f32>,
}

impl OperandScratch {
    pub(super) fn new(
        stream: &CudaStream,
        elements: usize,
        rows: usize,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, elements.div_ceil(2))?,
            scales: DeviceBuffer::zeroed(stream, elements.div_ceil(16))?,
            global_scales: DeviceBuffer::zeroed(stream, rows)?,
            chunk_amax: DeviceBuffer::zeroed(stream, elements.div_ceil(32))?,
        })
    }

    pub(super) fn operand(&mut self) -> MsEdenOperandScratch<'_> {
        MsEdenOperandScratch {
            bytes: &mut self.bytes,
            scales: &mut self.scales,
            global_scales: &mut self.global_scales,
            chunk_amax: &mut self.chunk_amax,
            global_scale: 1.0,
        }
    }
}
