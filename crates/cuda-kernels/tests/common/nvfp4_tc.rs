#![allow(dead_code)]

use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;
use rust_kernels_cuda::nvfp4_tc_matmul::{
    Nvfp4TcMatmulOperand, Nvfp4TcMatmulScratch, nvfp4_tc_matmul_bytes, nvfp4_tc_matmul_chunks,
    nvfp4_tc_matmul_elements, nvfp4_tc_matmul_scales,
};

pub struct Nvfp4TcScratchBuffers {
    a_padded: DeviceBuffer<f32>,
    b_t_padded: DeviceBuffer<f32>,
    a_bytes: DeviceBuffer<u8>,
    a_scales: DeviceBuffer<u8>,
    a_globals: DeviceBuffer<f32>,
    a_amax: DeviceBuffer<f32>,
    b_bytes: DeviceBuffer<u8>,
    b_scales: DeviceBuffer<u8>,
    b_globals: DeviceBuffer<f32>,
    b_amax: DeviceBuffer<f32>,
    a_global_scale: f32,
    b_global_scale: f32,
}

impl Nvfp4TcScratchBuffers {
    pub fn new(
        stream: &CudaStream,
        shape: (usize, usize, usize),
        global_scales: (f32, f32),
    ) -> Result<Self, DriverError> {
        let (m, n, k) = shape;
        let (m32, n32, k32) = (m as u32, n as u32, k as u32);
        Ok(Self {
            a_padded: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_elements(m32, k32))?,
            b_t_padded: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_elements(n32, k32))?,
            a_bytes: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_bytes(m32, k32))?,
            a_scales: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_scales(m32, k32))?,
            a_globals: DeviceBuffer::zeroed(stream, m)?,
            a_amax: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_chunks(m32, k32))?,
            b_bytes: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_bytes(n32, k32))?,
            b_scales: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_scales(n32, k32))?,
            b_globals: DeviceBuffer::zeroed(stream, n)?,
            b_amax: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_chunks(n32, k32))?,
            a_global_scale: global_scales.0,
            b_global_scale: global_scales.1,
        })
    }

    pub fn args(&mut self) -> Nvfp4TcMatmulScratch<'_> {
        Nvfp4TcMatmulScratch {
            a_padded: &mut self.a_padded,
            b_t_padded: &mut self.b_t_padded,
            a: operand(
                &mut self.a_bytes,
                &mut self.a_scales,
                &mut self.a_globals,
                &mut self.a_amax,
                self.a_global_scale,
            ),
            b_t: operand(
                &mut self.b_bytes,
                &mut self.b_scales,
                &mut self.b_globals,
                &mut self.b_amax,
                self.b_global_scale,
            ),
        }
    }

    pub fn a_tensor(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor::new(&self.a_bytes, &self.a_scales, &self.a_globals)
    }

    pub fn b_tensor(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor::new(&self.b_bytes, &self.b_scales, &self.b_globals)
    }
}

fn operand<'a>(
    bytes: &'a mut DeviceBuffer<u8>,
    scales: &'a mut DeviceBuffer<u8>,
    globals: &'a mut DeviceBuffer<f32>,
    amax: &'a mut DeviceBuffer<f32>,
    global_scale: f32,
) -> Nvfp4TcMatmulOperand<'a> {
    Nvfp4TcMatmulOperand {
        bytes,
        scales,
        global_scales: globals,
        chunk_amax: amax,
        global_scale,
    }
}
