mod runtime;
mod training;
mod upload;

use std::error::Error;

use training::{TokenDataLoader, Trainer};

type AppResult<T = ()> = Result<T, Box<dyn Error>>;

const SEED: u64 = 0x4750_5432;
const DEFAULT_TRAIN_STEPS: usize = 10;

fn main() -> AppResult {
    let mut trainer = Trainer::new(SEED)?;
    let mut data = TokenDataLoader::from_training_dataset()?;
    let mut previous_loss = None;
    let steps = train_steps();

    println!("training_tokens={} steps={steps}", data.token_count());

    for step in 0..steps {
        let window = data.next_window()?;
        let offset = window.offset;
        let source = window.source.display().to_string();
        let batch = trainer.batch_from_token_window(window.tokens)?;
        let stats = trainer.train_step(&batch)?;
        let delta = previous_loss
            .map(|loss| format!("{:+.6}", stats.loss - loss))
            .unwrap_or_else(|| "n/a".to_string());

        println!(
            "step={step} source={source} offset={offset} tokens={} logits={} loss={:.6} delta={} finite={} nonzero={} forward_ms={:.3} backward_enqueue_ms={:.3} loss_sync_ms={:.3} optimizer_ms={:.3} aurora_ms={:.3} adam_ms={:.3} embed_lookup_ms={:.3} token_embed_ms={:.3} final_norm_ms={:.3} blocks_ms={:.3}",
            stats.tokens,
            stats.logits,
            stats.loss,
            delta,
            stats.finite,
            stats.nonzero,
            stats.forward_ms,
            stats.backward_ms,
            stats.loss_sync_ms,
            stats.optimizer_ms,
            stats.optimizer.aurora_ms,
            stats.optimizer.adam_ms,
            stats.optimizer.embedding_lookup_ms,
            stats.optimizer.token_embedding_ms,
            stats.optimizer.final_norm_ms,
            stats.optimizer.blocks_ms,
        );
        previous_loss = Some(stats.loss);
    }

    Ok(())
}

fn train_steps() -> usize {
    std::env::var("TRAIN_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_TRAIN_STEPS)
}
