use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::nvfp4_tc_matmul::{
    Nvfp4TcMatmulOperand, Nvfp4TcMatmulScratch, nvfp4_tc_matmul_bytes, nvfp4_tc_matmul_chunks,
    nvfp4_tc_matmul_elements, nvfp4_tc_matmul_scales,
};
use rust_kernels_cuda::quartet;

pub struct Scratch {
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

impl Scratch {
    pub fn new(
        stream: &CudaStream,
        m: usize,
        n: usize,
        k: usize,
        a_global_scale: f32,
        b_global_scale: f32,
    ) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            a_padded: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_elements(m as u32, k as u32))?,
            b_t_padded: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_elements(n as u32, k as u32))?,
            a_bytes: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_bytes(m as u32, k as u32))?,
            a_scales: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_scales(m as u32, k as u32))?,
            a_globals: DeviceBuffer::zeroed(stream, m)?,
            a_amax: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_chunks(m as u32, k as u32))?,
            b_bytes: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_bytes(n as u32, k as u32))?,
            b_scales: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_scales(n as u32, k as u32))?,
            b_globals: DeviceBuffer::zeroed(stream, n)?,
            b_amax: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_chunks(n as u32, k as u32))?,
            a_global_scale,
            b_global_scale,
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
}

pub fn global_scale(x: &[f32]) -> f32 {
    let amax = x.iter().fold(0.0_f32, |acc, value| acc.max(value.abs()));
    quartet::quartet_backward_ms_eden_global_scale(amax)
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
