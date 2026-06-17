use cuda_core::{CudaStream, DeviceBuffer};
use gpt2_bpe::Gpt2Bpe;
use gpt2_nvfp4::GPT2_CONTEXT_LEN;

use crate::AppResult;

pub struct TokenBatch {
    pub tokens: DeviceBuffer<u32>,
    pub targets: DeviceBuffer<u32>,
    pub token_count: usize,
}

impl TokenBatch {
    pub fn from_text(stream: &CudaStream, text: &str) -> AppResult<Self> {
        let tokenizer = Gpt2Bpe::from_default_assets()?;
        let mut tokens = tokenizer.encode(text)?;
        tokens.truncate(GPT2_CONTEXT_LEN);
        let token_count = tokens.len();
        tokens.resize(GPT2_CONTEXT_LEN, tokenizer.eot_token());
        let mut targets = tokens[1..].to_vec();
        targets.push(tokenizer.eot_token());

        Ok(Self {
            tokens: DeviceBuffer::from_host(stream, &tokens)?,
            targets: DeviceBuffer::from_host(stream, &targets)?,
            token_count,
        })
    }
}
