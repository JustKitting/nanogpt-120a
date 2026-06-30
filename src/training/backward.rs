use std::time::Instant;

mod loss;
mod pass;
mod weights;

use super::{TokenBatch, TrainStats, Trainer};
use crate::AppResult;

impl Trainer {
    pub fn train_step(&mut self, batch: &TokenBatch, sync_loss: bool) -> AppResult<TrainStats> {
        super::schedule_free::materialize_training_weights(
            self.runtime.stream.as_ref(),
            &self.runtime,
            &mut self.uploaded,
            &mut self.buffers.optimizer,
            &self.buffers.optimizer_state,
        )?;

        let forward_start = Instant::now();
        let mut stats = self.forward_step(batch)?;
        stats.forward_ms = forward_start.elapsed().as_secs_f64() * 1000.0;

        let backward_start = Instant::now();
        self.enqueue_backward(batch)?;
        stats.backward_enqueue_ms = backward_start.elapsed().as_secs_f64() * 1000.0;

        if sync_loss {
            self.sync_loss(batch, &mut stats)?;
        }
        let observed_loss = sync_loss.then_some(stats.loss);

        let optimizer_start = Instant::now();
        let stream = self.runtime.stream.as_ref();
        let updates = super::optimizer_apply::apply_weight_updates(
            stream,
            &self.runtime,
            batch,
            &mut self.uploaded,
            &mut self.buffers.backward,
            &self.buffers.next_latent_grads,
            observed_loss,
            &mut self.buffers.optimizer,
            &mut self.buffers.optimizer_state,
            &mut self.buffers.aurora,
            &self.buffers.aurora_tables,
            &self.buffers.tape,
            &mut self.buffers.grad_clip,
        )?;
        stats.optimizer = updates.trace;
        stats.optimizer_ms = optimizer_start.elapsed().as_secs_f64() * 1000.0;
        stats.diagnostics = updates.diagnostics;

        Ok(stats)
    }
}
