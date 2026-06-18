use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD};
use rust_kernels_cuda::f16_tc_matmul::{F16TcMatmulScratch, f16_tc_matmul_elements};
use rust_kernels_cuda::optimizer::polar_normalize_chunks;

pub struct AuroraScratchBuffers {
    pub(super) update: DeviceBuffer<f32>,
    pub(super) oriented: DeviceBuffer<f32>,
    pub(super) scaled: DeviceBuffer<f32>,
    pub(super) u: DeviceBuffer<f32>,
    pub(super) polar_x: DeviceBuffer<f32>,
    pub(super) polar_chunks: DeviceBuffer<f32>,
    pub(super) polar_inv_norm: DeviceBuffer<f32>,
    pub(super) a: DeviceBuffer<f32>,
    pub(super) b: DeviceBuffer<f32>,
    pub(super) row_scale: DeviceBuffer<f32>,
    pub(super) tc: TcMatmulScratch,
}

pub(super) struct TcMatmulScratch {
    a_padded: DeviceBuffer<f32>,
    b_t_padded: DeviceBuffer<f32>,
    a_halves: DeviceBuffer<u16>,
    b_t_halves: DeviceBuffer<u16>,
}

impl AuroraScratchBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            update: DeviceBuffer::zeroed(stream, max_matrix())?,
            oriented: DeviceBuffer::zeroed(stream, max_matrix())?,
            scaled: DeviceBuffer::zeroed(stream, max_matrix())?,
            u: DeviceBuffer::zeroed(stream, max_matrix())?,
            polar_x: DeviceBuffer::zeroed(stream, max_matrix())?,
            polar_chunks: DeviceBuffer::zeroed(stream, polar_normalize_chunks(max_matrix()))?,
            polar_inv_norm: DeviceBuffer::zeroed(stream, 1)?,
            a: DeviceBuffer::zeroed(stream, small_square())?,
            b: DeviceBuffer::zeroed(stream, small_square())?,
            row_scale: DeviceBuffer::zeroed(stream, GPT2_MLP)?,
            tc: TcMatmulScratch::new(stream)?,
        })
    }
}

impl TcMatmulScratch {
    fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        let matmul = f16_tc_matmul_elements(max_dim(), max_dim());
        Ok(Self {
            a_padded: DeviceBuffer::zeroed(stream, matmul)?,
            b_t_padded: DeviceBuffer::zeroed(stream, matmul)?,
            a_halves: DeviceBuffer::zeroed(stream, matmul)?,
            b_t_halves: DeviceBuffer::zeroed(stream, matmul)?,
        })
    }

    pub(super) fn scratch(&mut self) -> F16TcMatmulScratch<'_> {
        F16TcMatmulScratch {
            a_padded: &mut self.a_padded,
            b_t_padded: &mut self.b_t_padded,
            a_halves: &mut self.a_halves,
            b_t_halves: &mut self.b_t_halves,
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
    max_dim() as usize * max_dim() as usize
}
