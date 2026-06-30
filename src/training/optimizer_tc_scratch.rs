use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD, GPT2_N_LAYER, GPT2_QKV, NEXTLAT_HIDDEN, NEXTLAT_INPUT};
use rust_kernels_cuda::optimizer::{AURORA_COOPERATIVE_BLOCKS, AURORA_MATRIX_PHASES};

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
            oriented: DeviceBuffer::zeroed(stream, grouped(max_matrix()))?,
            polar_next: DeviceBuffer::zeroed(stream, grouped(max_matrix()))?,
            polar_x: DeviceBuffer::zeroed(stream, grouped(max_matrix()))?,
            polar_gram: DeviceBuffer::zeroed(stream, grouped(max_gram() * max_gram()))?,
            polar_ax: DeviceBuffer::zeroed(stream, grouped(max_square_matrix()))?,
            polar_chunks: DeviceBuffer::zeroed(stream, grouped(AURORA_COOPERATIVE_BLOCKS))?,
        })
    }
}

const fn grouped(len: usize) -> usize {
    len * active_matrix_slots()
}

const AURORA_MATRIX_SLOTS: usize = GPT2_N_LAYER * 4 + 3;

const fn active_matrix_slots() -> usize {
    AURORA_MATRIX_SLOTS.div_ceil(AURORA_MATRIX_PHASES)
}

const fn max_matrix() -> usize {
    max3(
        GPT2_N_EMBD * GPT2_QKV,
        GPT2_MLP * GPT2_N_EMBD,
        NEXTLAT_INPUT * NEXTLAT_HIDDEN,
    )
}

const fn max_square_matrix() -> usize {
    max2(GPT2_N_EMBD * GPT2_N_EMBD, NEXTLAT_HIDDEN * NEXTLAT_HIDDEN)
}

const fn max_gram() -> usize {
    max2(GPT2_N_EMBD, NEXTLAT_HIDDEN)
}

const fn max2(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

const fn max3(a: usize, b: usize, c: usize) -> usize {
    max2(max2(a, b), c)
}
