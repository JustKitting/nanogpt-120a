use cuda_core::DeviceBuffer;
use gpt2_bpe::Gpt2Bpe;
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_VOCAB_SIZE};
use rust_kernels_cuda::logits::LogitsArgmaxArgs;

use super::{TokenBatch, Trainer};
use crate::AppResult;

impl Trainer {
    pub fn generate_greedy(&mut self, prompt: &str, max_new_tokens: usize) -> AppResult<String> {
        let tokenizer = Gpt2Bpe::from_default_assets()?;
        let mut tokens = tokenizer.encode(prompt)?;
        if tokens.is_empty() {
            tokens.push(tokenizer.eot_token());
        }

        let stream = self.runtime.stream.clone();
        let mut next_token_dev = DeviceBuffer::<u32>::zeroed(stream.as_ref(), 1)?;

        for _ in 0..max_new_tokens {
            let (windows, row) = generation_batch(&tokens, tokenizer.eot_token())?;
            let batch = TokenBatch::from_default_batch(stream.as_ref(), &windows)?;
            self.forward_step(&batch)?;
            self.runtime.logits.argmax(LogitsArgmaxArgs {
                stream: stream.as_ref(),
                logits: &self.buffers.logits,
                out_token: &mut next_token_dev,
                row,
                vocab_size: GPT2_VOCAB_SIZE as u32,
            })?;
            let next = next_token_dev.to_host_vec(stream.as_ref())?[0];
            tokens.push(next);
        }

        tokenizer.decode(&tokens)
    }
}

fn generation_batch(tokens: &[u32], pad_token: u32) -> AppResult<(Vec<u16>, u32)> {
    let context_len = tokens.len().min(GPT2_SEQ_LEN);
    let context_start = tokens.len() - context_len;
    let row = context_len.saturating_sub(1) as u32;
    let window_len = GPT2_SEQ_LEN + 1;
    let pad = u16::try_from(pad_token)?;
    let mut one = vec![pad; window_len];

    for (dst, &token) in one.iter_mut().zip(tokens[context_start..].iter()) {
        *dst = u16::try_from(token)?;
    }

    let mut windows = Vec::with_capacity(GPT2_BATCH_SIZE * window_len);
    for _ in 0..GPT2_BATCH_SIZE {
        windows.extend_from_slice(&one);
    }

    Ok((windows, row))
}
