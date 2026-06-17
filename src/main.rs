mod runtime;
mod training;
mod upload;

use std::error::Error;

use training::Trainer;

type AppResult<T = ()> = Result<T, Box<dyn Error>>;

const SEED: u64 = 0x4750_5432;
const SAMPLE_TEXT: &str = "NVFP4 GPT-2 forward smoke test.";

fn main() -> AppResult {
    let mut trainer = Trainer::new(SEED)?;
    let batch = trainer.batch_from_text(SAMPLE_TEXT)?;

    for step in 0..train_steps() {
        let stats = trainer.train_step(&batch)?;
        println!(
            "step={step} tokens={} logits={} loss={:.6} finite={} nonzero={}",
            stats.tokens, stats.logits, stats.loss, stats.finite, stats.nonzero
        );
    }

    Ok(())
}

fn train_steps() -> usize {
    std::env::var("TRAIN_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1)
}
