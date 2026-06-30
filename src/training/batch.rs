use cuda_core::{CudaStream, DeviceBuffer};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};

use crate::AppResult;

mod reusable;
mod windows;

pub use reusable::ReusableTokenBatch;

pub struct TokenBatch {
    pub tokens: DeviceBuffer<u32>,
    pub targets: DeviceBuffer<u32>,
    pub batch_size: usize,
    pub seq_len: usize,
    pub token_count: usize,
}

impl TokenBatch {
    pub fn from_default_batch(stream: &CudaStream, windows: &[u16]) -> AppResult<Self> {
        Self::from_flat_windows(stream, windows, GPT2_BATCH_SIZE, GPT2_SEQ_LEN)
    }

    pub fn from_flat_windows(
        stream: &CudaStream,
        windows: &[u16],
        batch_size: usize,
        seq_len: usize,
    ) -> AppResult<Self> {
        let token_count = batch_size * seq_len;
        let mut tokens = vec![0; token_count];
        let mut targets = vec![0; token_count];
        windows::fill(windows, batch_size, seq_len, &mut tokens, &mut targets)?;

        Ok(Self {
            tokens: DeviceBuffer::from_host(stream, &tokens)?,
            targets: DeviceBuffer::from_host(stream, &targets)?,
            batch_size,
            seq_len,
            token_count,
        })
    }
}
