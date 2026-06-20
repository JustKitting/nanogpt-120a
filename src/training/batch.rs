use cuda_core::{CudaEvent, CudaStream, DeviceBuffer, PinnedHostBuffer};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};

use crate::AppResult;

pub struct TokenBatch {
    pub tokens: DeviceBuffer<u32>,
    pub targets: DeviceBuffer<u32>,
    pub batch_size: usize,
    pub seq_len: usize,
    pub token_count: usize,
}

pub struct ReusableTokenBatch {
    batch: TokenBatch,
    host_tokens: PinnedHostBuffer<u32>,
    host_targets: PinnedHostBuffer<u32>,
    copy_done: CudaEvent,
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

impl ReusableTokenBatch {
    pub fn default(stream: &CudaStream) -> AppResult<Self> {
        Self::new(stream, GPT2_BATCH_SIZE, GPT2_SEQ_LEN)
    }

    pub fn new(stream: &CudaStream, batch_size: usize, seq_len: usize) -> AppResult<Self> {
        let token_count = batch_size * seq_len;
        let copy_done = stream.context().new_event(None)?;
        copy_done.record(stream)?;

        Ok(Self {
            batch: TokenBatch {
                tokens: DeviceBuffer::zeroed(stream, token_count)?,
                targets: DeviceBuffer::zeroed(stream, token_count)?,
                batch_size,
                seq_len,
                token_count,
            },
            host_tokens: PinnedHostBuffer::zeroed(stream.context(), token_count)?,
            host_targets: PinnedHostBuffer::zeroed(stream.context(), token_count)?,
            copy_done,
        })
    }

    pub fn upload(&mut self, stream: &CudaStream, windows: &[u16]) -> AppResult<&TokenBatch> {
        self.copy_done.synchronize()?;
        fill_host_windows(
            windows,
            self.batch.batch_size,
            self.batch.seq_len,
            self.host_tokens.as_mut_slice(),
            self.host_targets.as_mut_slice(),
        )?;

        unsafe {
            self.batch
                .tokens
                .copy_from_pinned_host_async(stream, &self.host_tokens)?;
            self.batch
                .targets
                .copy_from_pinned_host_async(stream, &self.host_targets)?;
        }
        self.copy_done.record(stream)?;
        Ok(&self.batch)
    }
}

fn fill_host_windows(
    windows: &[u16],
    batch_size: usize,
    seq_len: usize,
    tokens: &mut [u32],
    targets: &mut [u32],
) -> AppResult {
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

    for batch in 0..batch_size {
        let window_base = batch * window_len;
        let out_base = batch * seq_len;
        for col in 0..seq_len {
            tokens[out_base + col] = windows[window_base + col] as u32;
            targets[out_base + col] = windows[window_base + col + 1] as u32;
        }
    }

    Ok(())
}
