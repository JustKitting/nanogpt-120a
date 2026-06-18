use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD};
use rust_kernels_cuda::f16_tc_matmul::{f16_tc_matmul_elements, f16_tc_matmul_padded_k};
use rust_kernels_cuda::optimizer::polar_normalize_chunks;

pub struct AuroraScratchBuffers {
    pub(super) update: DeviceBuffer<f32>,
    pub(super) oriented: DeviceBuffer<f32>,
    pub(super) scaled: DeviceBuffer<f32>,
    pub(super) u: DeviceBuffer<f32>,
    pub(super) polar_x: DeviceBuffer<f32>,
    pub(super) polar_gram: DeviceBuffer<f32>,
    pub(super) polar_ax: DeviceBuffer<f32>,
    pub(super) polar_chunks: DeviceBuffer<f32>,
    pub(super) polar_inv_norm: DeviceBuffer<f32>,
    pub(super) row_scale: DeviceBuffer<f32>,
    pub(super) a_padded: DeviceBuffer<f32>,
    pub(super) b_t_padded: DeviceBuffer<f32>,
    pub(super) a_halves: DeviceBuffer<u16>,
    pub(super) b_t_halves: DeviceBuffer<u16>,
}

impl AuroraScratchBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            update: DeviceBuffer::zeroed(stream, max_matrix())?,
            oriented: DeviceBuffer::zeroed(stream, max_matrix())?,
            scaled: DeviceBuffer::zeroed(stream, max_matrix())?,
            u: DeviceBuffer::zeroed(stream, max_matrix())?,
            polar_x: DeviceBuffer::zeroed(stream, max_matrix())?,
            polar_gram: DeviceBuffer::zeroed(stream, max_rows() * max_rows())?,
            polar_ax: DeviceBuffer::zeroed(stream, max_matrix())?,
            polar_chunks: DeviceBuffer::zeroed(stream, polar_normalize_chunks(max_matrix()))?,
            polar_inv_norm: DeviceBuffer::zeroed(stream, 1)?,
            row_scale: DeviceBuffer::zeroed(stream, GPT2_MLP)?,
            a_padded: DeviceBuffer::zeroed(stream, tc_scratch_len())?,
            b_t_padded: DeviceBuffer::zeroed(stream, tc_scratch_len())?,
            a_halves: DeviceBuffer::zeroed(stream, tc_scratch_len())?,
            b_t_halves: DeviceBuffer::zeroed(stream, tc_scratch_len())?,
        })
    }
}

const fn max_matrix() -> usize {
    GPT2_MLP * GPT2_N_EMBD
}

const fn max_rows() -> usize {
    GPT2_N_EMBD
}

fn tc_scratch_len() -> usize {
    f16_tc_matmul_elements(max_rows() as u32, max_cols())
}

fn max_cols() -> u32 {
    f16_tc_matmul_padded_k(GPT2_MLP as u32)
}
