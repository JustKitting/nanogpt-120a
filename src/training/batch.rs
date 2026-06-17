use cuda_core::{CudaStream, DeviceBuffer};
use gpt2_nvfp4::GPT2_CONTEXT_LEN;

use crate::AppResult;

pub struct TokenBatch {
    pub tokens: DeviceBuffer<u32>,
    pub targets: DeviceBuffer<u32>,
    pub token_count: usize,
}

impl TokenBatch {
    pub fn from_token_window(stream: &CudaStream, window: &[u16]) -> AppResult<Self> {
        if window.len() < GPT2_CONTEXT_LEN + 1 {
            return Err(format!(
                "token window has {} tokens, needs {}",
                window.len(),
                GPT2_CONTEXT_LEN + 1
            )
            .into());
        }

        let tokens: Vec<u32> = window[..GPT2_CONTEXT_LEN]
            .iter()
            .map(|&id| id as u32)
            .collect();
        let targets: Vec<u32> = window[1..=GPT2_CONTEXT_LEN]
            .iter()
            .map(|&id| id as u32)
            .collect();

        Ok(Self {
            tokens: DeviceBuffer::from_host(stream, &tokens)?,
            targets: DeviceBuffer::from_host(stream, &targets)?,
            token_count: GPT2_CONTEXT_LEN,
        })
    }
}
