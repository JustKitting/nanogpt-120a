#![allow(dead_code)]

use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::f16_tc_matmul::{f16_tc_matmul_elements, F16TcMatmulScratch};

pub struct F16TcScratchBuffers {
    a_padded: DeviceBuffer<f32>,
    b_t_padded: DeviceBuffer<f32>,
    a_halves: DeviceBuffer<u16>,
    b_t_halves: DeviceBuffer<u16>,
}

impl F16TcScratchBuffers {
    pub fn new(stream: &CudaStream, shape: (usize, usize, usize)) -> Result<Self, DriverError> {
        let (a_rows, b_rows, k) = shape;
        let a_len = f16_tc_matmul_elements(a_rows as u32, k as u32);
        let b_len = f16_tc_matmul_elements(b_rows as u32, k as u32);
        Ok(Self {
            a_padded: DeviceBuffer::zeroed(stream, a_len)?,
            b_t_padded: DeviceBuffer::zeroed(stream, b_len)?,
            a_halves: DeviceBuffer::zeroed(stream, a_len)?,
            b_t_halves: DeviceBuffer::zeroed(stream, b_len)?,
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
