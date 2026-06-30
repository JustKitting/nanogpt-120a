use cuda_core::{CudaEvent, CudaStream, DeviceBuffer, PinnedHostBuffer};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};

use super::{TokenBatch, windows};
use crate::AppResult;

pub struct ReusableTokenBatch {
    batch: TokenBatch,
    host_tokens: PinnedHostBuffer<u32>,
    host_targets: PinnedHostBuffer<u32>,
    copy_done: CudaEvent,
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

    pub fn upload(&mut self, stream: &CudaStream, token_windows: &[u16]) -> AppResult<&TokenBatch> {
        self.copy_done.synchronize()?;
        windows::fill(
            token_windows,
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
