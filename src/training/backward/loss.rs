use std::time::Instant;

use crate::{
    AppResult,
    training::{TokenBatch, TrainStats, Trainer},
};

impl Trainer {
    pub(super) fn sync_loss(&mut self, batch: &TokenBatch, stats: &mut TrainStats) -> AppResult {
        let loss_sync_start = Instant::now();
        let losses = self
            .buffers
            .backward
            .losses
            .to_host_vec(self.runtime.stream.as_ref())?;
        let active_losses = &losses[..batch.token_count];
        stats.loss = active_losses.iter().sum::<f32>() / active_losses.len() as f32;
        stats.finite &= active_losses.iter().all(|value| value.is_finite());
        stats.nonzero |= active_losses.iter().any(|value| value.abs() > 0.0);
        stats.loss_host_wait_ms = loss_sync_start.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
}
