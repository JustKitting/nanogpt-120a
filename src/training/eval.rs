use gpt2_nvfp4::GPT2_VOCAB_SIZE;
use rust_kernels_cuda::loss::CrossEntropyArgs;

use super::{TokenBatch, Trainer};
use crate::AppResult;

impl Trainer {
    pub fn eval_loss(&mut self, batch: &TokenBatch) -> AppResult<f32> {
        self.forward_step(batch)?;

        let stream = self.runtime.stream.as_ref();
        self.runtime.loss.cross_entropy(CrossEntropyArgs {
            stream,
            logits: &self.buffers.logits,
            targets: &batch.targets,
            losses: &mut self.buffers.backward.losses,
            dlogits: &mut self.buffers.backward.dlogits,
            token_count: batch.token_count as u32,
            vocab_size: GPT2_VOCAB_SIZE as u32,
        })?;

        let losses = self.buffers.backward.losses.to_host_vec(stream)?;
        let active_losses = &losses[..batch.token_count];
        Ok(active_losses.iter().sum::<f32>() / active_losses.len() as f32)
    }
}
