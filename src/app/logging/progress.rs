use crate::loss_graph::LossCurve;
use crate::training::TrainStats;

pub struct TrainingLogger {
    previous_loss: Option<f32>,
    loss_ema: Option<f32>,
    loss_curve: LossCurve,
}

pub struct StepLogContext<'a> {
    pub step: usize,
    pub source: &'a str,
    pub offset: usize,
    pub batch_size: usize,
    pub seq_len: usize,
}

impl TrainingLogger {
    pub fn new() -> Self {
        Self {
            previous_loss: None,
            loss_ema: None,
            loss_curve: LossCurve::new(),
        }
    }

    pub fn loss_curve(&self) -> &LossCurve {
        &self.loss_curve
    }

    pub fn log_step(&mut self, context: StepLogContext<'_>, stats: &TrainStats) {
        let step = context.step;
        let delta = self
            .previous_loss
            .map(|loss| format!("{:+.6}", stats.loss - loss))
            .unwrap_or_else(|| "n/a".to_string());
        let ema = self.update_loss_ema(stats.loss);
        self.loss_curve.push(step, stats.loss, ema);
        println!(
            "step={step} source={} offset={} batch_size={} seq_len={} tokens={} logits={} loss={:.6} loss_ema={:.6} delta={} finite={} nonzero={} adam_lr={:.6e} aurora_lr={:.6e} forward_ms={:.3} backward_enqueue_ms={:.3} loss_sync_ms={:.3} optimizer_ms={:.3} aurora_ms={:.3} adam_ms={:.3} embed_lookup_ms={:.3} token_embed_ms={:.3} final_norm_ms={:.3} blocks_ms={:.3}",
            context.source,
            context.offset,
            context.batch_size,
            context.seq_len,
            stats.tokens,
            stats.logits,
            stats.loss,
            ema,
            delta,
            stats.finite,
            stats.nonzero,
            stats.optimizer.adam_lr,
            stats.optimizer.aurora_lr,
            stats.forward_ms,
            stats.backward_enqueue_ms,
            stats.loss_sync_ms,
            stats.optimizer_ms,
            stats.optimizer.aurora_ms,
            stats.optimizer.adam_ms,
            stats.optimizer.embedding_lookup_ms,
            stats.optimizer.token_embedding_ms,
            stats.optimizer.final_norm_ms,
            stats.optimizer.blocks_ms,
        );
        self.previous_loss = Some(stats.loss);
    }

    fn update_loss_ema(&mut self, loss: f32) -> f32 {
        const BETA: f32 = 0.9;
        let next = self
            .loss_ema
            .map(|ema| BETA * ema + (1.0 - BETA) * loss)
            .unwrap_or(loss);
        self.loss_ema = Some(next);
        next
    }
}
