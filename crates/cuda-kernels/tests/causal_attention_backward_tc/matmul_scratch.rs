use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::f16_tc_matmul::{F16TcMatmulScratch, f16_tc_matmul_elements};

use super::shape::{HEAD_DIM, TOKEN_COUNT};

pub struct TcMatmulScratchBuffers {
    a_padded: DeviceBuffer<f32>,
    b_t_padded: DeviceBuffer<f32>,
    a_halves: DeviceBuffer<u16>,
    b_t_halves: DeviceBuffer<u16>,
}

impl TcMatmulScratchBuffers {
    pub fn new(stream: &CudaStream, rows: usize) -> Result<Self, DriverError> {
        Ok(Self {
            a_padded: zero(stream, elements(rows))?,
            b_t_padded: zero(stream, elements(rows))?,
            a_halves: zero(stream, elements(rows))?,
            b_t_halves: zero(stream, elements(rows))?,
        })
    }

    pub fn args(&mut self) -> F16TcMatmulScratch<'_> {
        F16TcMatmulScratch {
            a_padded: &mut self.a_padded,
            b_t_padded: &mut self.b_t_padded,
            a_halves: &mut self.a_halves,
            b_t_halves: &mut self.b_t_halves,
        }
    }
}

fn zero<T: cuda_core::DeviceCopy>(
    stream: &CudaStream,
    len: usize,
) -> Result<DeviceBuffer<T>, DriverError> {
    DeviceBuffer::zeroed(stream, len)
}

fn elements(rows: usize) -> usize {
    f16_tc_matmul_elements(rows as u32, TOKEN_COUNT.max(HEAD_DIM) as u32)
}
