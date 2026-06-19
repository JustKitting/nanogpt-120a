use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_VOCAB_SIZE};
use llama2_tokenizer::Llama2Tokenizer;
use rust_kernels_cuda::logits::{LOGITS_TOP_K, LogitsTopKArgs};

use super::{TokenBatch, Trainer};
use crate::AppResult;

#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Trainer {
    pub fn generate_sampled(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        config: SamplingConfig,
    ) -> AppResult<String> {
        let tokenizer = Llama2Tokenizer::from_default_assets()?;
        let mut tokens = tokenizer.encode(prompt)?;
        if tokens.is_empty() {
            tokens.push(tokenizer.bos_token());
        }

        let stream = self.runtime.stream.clone();
        let top_k = config.top_k.clamp(1, LOGITS_TOP_K);
        let mut top_tokens_dev = DeviceBuffer::<u32>::zeroed(stream.as_ref(), top_k)?;
        let mut top_logits_dev = DeviceBuffer::<f32>::zeroed(stream.as_ref(), top_k)?;

        for _ in 0..max_new_tokens {
            let (windows, row) = generation_batch(&tokens, tokenizer.eos_token())?;
            let batch = TokenBatch::from_default_batch(stream.as_ref(), &windows)?;
            self.forward_step(&batch)?;
            self.runtime.logits.top_k(LogitsTopKArgs {
                stream: stream.as_ref(),
                logits: &self.buffers.logits,
                out_tokens: &mut top_tokens_dev,
                out_values: &mut top_logits_dev,
                row,
                vocab_size: GPT2_VOCAB_SIZE as u32,
                k: top_k as u32,
            })?;
            let top_tokens = top_tokens_dev.to_host_vec(stream.as_ref())?;
            let top_logits = top_logits_dev.to_host_vec(stream.as_ref())?;
            let next = sample_top_k(
                &top_tokens,
                &top_logits,
                config.temperature,
                config.top_p,
                &mut self.rng,
            );
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

fn sample_top_k(
    tokens: &[u32],
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    rng: &mut gpt2_nvfp4::Gpt2Rng,
) -> u32 {
    let temperature = if temperature.is_finite() && temperature > 0.0 {
        temperature
    } else {
        1.0
    };
    let top_p = if top_p.is_finite() && top_p > 0.0 {
        top_p.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |max, value| max.max(value));
    let mut weights = Vec::with_capacity(logits.len());
    let mut total = 0.0_f64;

    for &logit in logits {
        let weight = ((logit - max_logit) / temperature).exp() as f64;
        let weight = if weight.is_finite() { weight } else { 0.0 };
        weights.push(weight);
        total += weight;
    }

    if total <= 0.0 || !total.is_finite() {
        return tokens[0];
    }

    let sample_total = nucleus_total(&weights, total, top_p);
    let mut draw = uniform01(rng) * sample_total;
    for (&token, weight) in tokens.iter().zip(weights) {
        if draw <= weight {
            return token;
        }
        draw -= weight;
    }
    tokens[0]
}

fn nucleus_total(weights: &[f64], total: f64, top_p: f32) -> f64 {
    if top_p >= 1.0 {
        return total;
    }

    let cutoff = total * top_p as f64;
    let mut selected = 0.0_f64;
    for &weight in weights {
        selected += weight;
        if selected >= cutoff {
            return selected.max(weight);
        }
    }
    selected.max(weights.first().copied().unwrap_or(total))
}

fn uniform01(rng: &mut gpt2_nvfp4::Gpt2Rng) -> f64 {
    (rng.next_u32() as f64 + 0.5) / (u32::MAX as f64 + 1.0)
}
