use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::optimizer::{AURORA_COOPERATIVE_BLOCKS, AURORA_MATRIX_PHASES};

use super::optimizer_aurora::{
    AURORA_MATRIX_SLOTS, max_matrix_dim, max_matrix_len, max_polar_ax_len,
};

pub struct AuroraScratchBuffers {
    pub(super) oriented: DeviceBuffer<f32>,
    pub(super) polar_next: DeviceBuffer<f32>,
    pub(super) polar_x: DeviceBuffer<f32>,
    pub(super) polar_gram: DeviceBuffer<f32>,
    pub(super) polar_ax: DeviceBuffer<f32>,
    pub(super) polar_chunks: DeviceBuffer<f32>,
}

impl AuroraScratchBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            oriented: DeviceBuffer::zeroed(stream, grouped(max_matrix_len()))?,
            polar_next: DeviceBuffer::zeroed(stream, grouped(max_matrix_len()))?,
            polar_x: DeviceBuffer::zeroed(stream, grouped(max_matrix_len()))?,
            polar_gram: DeviceBuffer::zeroed(stream, grouped(max_matrix_dim() * max_matrix_dim()))?,
            polar_ax: DeviceBuffer::zeroed(stream, grouped(max_polar_ax_len()))?,
            polar_chunks: DeviceBuffer::zeroed(stream, grouped(AURORA_COOPERATIVE_BLOCKS))?,
        })
    }
}

const fn grouped(len: usize) -> usize {
    len * active_matrix_slots()
}

const fn active_matrix_slots() -> usize {
    AURORA_MATRIX_SLOTS.div_ceil(AURORA_MATRIX_PHASES)
}
