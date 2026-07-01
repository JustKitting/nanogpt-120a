use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;
use rust_kernels_cuda::nvfp4_tc_matmul::{
    Nvfp4TcMatmulOperand, Nvfp4TcMatmulScratch, nvfp4_tc_matmul_bytes, nvfp4_tc_matmul_chunks,
    nvfp4_tc_matmul_elements, nvfp4_tc_matmul_padded_k, nvfp4_tc_matmul_scales,
};

pub const M: usize = 1;
pub const N: usize = 1;
pub const K: usize = 65;

pub struct ScratchBuffers {
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
}

impl ScratchBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            a_padded: DeviceBuffer::zeroed(stream, elements(M))?,
            b_t_padded: DeviceBuffer::zeroed(stream, elements(N))?,
            a_bytes: DeviceBuffer::zeroed(stream, bytes(M))?,
            a_scales: DeviceBuffer::zeroed(stream, scales(M))?,
            a_globals: DeviceBuffer::zeroed(stream, M)?,
            a_amax: DeviceBuffer::zeroed(stream, chunks(M))?,
            b_bytes: DeviceBuffer::zeroed(stream, bytes(N))?,
            b_scales: DeviceBuffer::zeroed(stream, scales(N))?,
            b_globals: DeviceBuffer::zeroed(stream, N)?,
            b_amax: DeviceBuffer::zeroed(stream, chunks(N))?,
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
            ),
            b_t: operand(
                &mut self.b_bytes,
                &mut self.b_scales,
                &mut self.b_globals,
                &mut self.b_amax,
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
) -> Nvfp4TcMatmulOperand<'a> {
    Nvfp4TcMatmulOperand {
        bytes,
        scales,
        global_scales: globals,
        chunk_amax: amax,
        global_scale: 1.0,
    }
}

pub fn padded_k() -> usize {
    nvfp4_tc_matmul_padded_k(K as u32) as usize
}

fn elements(rows: usize) -> usize {
    nvfp4_tc_matmul_elements(rows as u32, K as u32)
}

fn bytes(rows: usize) -> usize {
    nvfp4_tc_matmul_bytes(rows as u32, K as u32)
}

fn scales(rows: usize) -> usize {
    nvfp4_tc_matmul_scales(rows as u32, K as u32)
}

fn chunks(rows: usize) -> usize {
    nvfp4_tc_matmul_chunks(rows as u32, K as u32)
}
