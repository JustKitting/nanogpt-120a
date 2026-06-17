use cuda_core::{CudaStream, DeviceBuffer};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};

use crate::AppResult;

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
        let window_len = seq_len + 1;
        let needed = batch_size * window_len;
        if windows.len() < needed {
            return Err(format!(
                "token window has {} tokens, needs {}",
                windows.len(),
                needed
            )
            .into());
        }

        let mut tokens = Vec::with_capacity(batch_size * seq_len);
        let mut targets = Vec::with_capacity(batch_size * seq_len);
        for batch in 0..batch_size {
            let base = batch * window_len;
            tokens.extend(windows[base..base + seq_len].iter().map(|&id| id as u32));
            targets.extend(
                windows[base + 1..base + window_len]
                    .iter()
                    .map(|&id| id as u32),
            );
        }

        Ok(Self {
            tokens: DeviceBuffer::from_host(stream, &tokens)?,
            targets: DeviceBuffer::from_host(stream, &targets)?,
            batch_size,
            seq_len,
            token_count: batch_size * seq_len,
        })
    }
}
