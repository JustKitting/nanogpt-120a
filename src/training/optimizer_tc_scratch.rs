use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD};
use rust_kernels_cuda::nvfp4_tc_matmul::{
    Nvfp4TcMatmulOperand, Nvfp4TcMatmulScratch, nvfp4_tc_matmul_bytes, nvfp4_tc_matmul_chunks,
    nvfp4_tc_matmul_elements, nvfp4_tc_matmul_scales,
};

pub struct AuroraScratchBuffers {
    pub(super) update: DeviceBuffer<f32>,
    pub(super) oriented: DeviceBuffer<f32>,
    pub(super) scaled: DeviceBuffer<f32>,
    pub(super) u: DeviceBuffer<f32>,
    pub(super) polar_x: DeviceBuffer<f32>,
    pub(super) polar_xt: DeviceBuffer<f32>,
    pub(super) polar_next: DeviceBuffer<f32>,
    pub(super) a: DeviceBuffer<f32>,
    pub(super) a_t: DeviceBuffer<f32>,
    pub(super) aa: DeviceBuffer<f32>,
    pub(super) b: DeviceBuffer<f32>,
    pub(super) bx: DeviceBuffer<f32>,
    pub(super) norm: DeviceBuffer<f32>,
    pub(super) row_scale: DeviceBuffer<f32>,
    pub(super) tc: TcMatmulScratch,
}

pub(super) struct TcMatmulScratch {
    a_padded: DeviceBuffer<f32>,
    b_t_padded: DeviceBuffer<f32>,
    a: TcOperand,
    b_t: TcOperand,
}

struct TcOperand {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scales: DeviceBuffer<f32>,
    chunk_amax: DeviceBuffer<f32>,
}

impl AuroraScratchBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            update: DeviceBuffer::zeroed(stream, max_matrix())?,
            oriented: DeviceBuffer::zeroed(stream, max_matrix())?,
            scaled: DeviceBuffer::zeroed(stream, max_matrix())?,
            u: DeviceBuffer::zeroed(stream, max_matrix())?,
            polar_x: DeviceBuffer::zeroed(stream, max_matrix())?,
            polar_xt: DeviceBuffer::zeroed(stream, max_matrix())?,
            polar_next: DeviceBuffer::zeroed(stream, max_matrix())?,
            a: DeviceBuffer::zeroed(stream, small_square())?,
            a_t: DeviceBuffer::zeroed(stream, small_square())?,
            aa: DeviceBuffer::zeroed(stream, small_square())?,
            b: DeviceBuffer::zeroed(stream, small_square())?,
            bx: DeviceBuffer::zeroed(stream, max_matrix())?,
            norm: DeviceBuffer::zeroed(stream, 1)?,
            row_scale: DeviceBuffer::zeroed(stream, GPT2_MLP)?,
            tc: TcMatmulScratch::new(stream)?,
        })
    }
}

impl TcMatmulScratch {
    fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            a_padded: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_elements(max_dim(), max_dim()))?,
            b_t_padded: DeviceBuffer::zeroed(
                stream,
                nvfp4_tc_matmul_elements(max_dim(), max_dim()),
            )?,
            a: TcOperand::new(stream)?,
            b_t: TcOperand::new(stream)?,
        })
    }

    pub(super) fn scratch(&mut self) -> Nvfp4TcMatmulScratch<'_> {
        Nvfp4TcMatmulScratch {
            a_padded: &mut self.a_padded,
            b_t_padded: &mut self.b_t_padded,
            a: self.a.operand(),
            b_t: self.b_t.operand(),
        }
    }
}

impl TcOperand {
    fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_bytes(max_dim(), max_dim()))?,
            scales: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_scales(max_dim(), max_dim()))?,
            global_scales: DeviceBuffer::zeroed(stream, max_dim() as usize)?,
            chunk_amax: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_chunks(max_dim(), max_dim()))?,
        })
    }

    fn operand(&mut self) -> Nvfp4TcMatmulOperand<'_> {
        Nvfp4TcMatmulOperand {
            bytes: &mut self.bytes,
            scales: &mut self.scales,
            global_scales: &mut self.global_scales,
            chunk_amax: &mut self.chunk_amax,
            global_scale: 1.0,
        }
    }
}

const fn max_dim() -> u32 {
    GPT2_MLP as u32
}

const fn max_matrix() -> usize {
    GPT2_MLP * GPT2_N_EMBD
}

const fn small_square() -> usize {
    GPT2_N_EMBD * GPT2_N_EMBD
}
