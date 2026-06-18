mod checkpoint;
mod loss_graph;
mod runtime;
mod training;
mod upload;

use std::error::Error;
use std::path::PathBuf;

use training::{TokenDataLoader, Trainer};

type AppResult<T = ()> = Result<T, Box<dyn Error>>;

const SEED: u64 = 0x4750_5432;
const DEFAULT_TRAIN_STEPS: usize = 10;

fn main() -> AppResult {
    let mut trainer = Trainer::new(SEED)?;
    let mut data = TokenDataLoader::from_training_dataset()?;
    let mut previous_loss = None;
    let mut loss_ema = None;
    let steps = train_steps();
    let log_interval = train_log_interval();
    let eval_interval = train_eval_interval();
    let mut loss_curve = loss_graph::LossCurve::new();
    let validation_tokens = data.validation_tokens()?;
    let validation_batch = trainer.batch_from_default_windows(&validation_tokens)?;

    println!("training_tokens={} steps={steps}", data.token_count());

    for step in 0..steps {
        let log_step = should_log_step(step, steps, log_interval);
        let window = data.next_batch()?;
        let offset = window.offset;
        let source = window.source.display().to_string();
        let window_batch_size = window.batch_size;
        let window_seq_len = window.seq_len;
        let batch = trainer.batch_from_default_windows(&window.tokens)?;
        let stats = trainer.train_step(&batch, log_step)?;

        if log_step {
            let delta = previous_loss
                .map(|loss| format!("{:+.6}", stats.loss - loss))
                .unwrap_or_else(|| "n/a".to_string());
            let ema = update_loss_ema(&mut loss_ema, stats.loss);
            loss_curve.push(step, stats.loss, ema);
            println!(
                "step={step} source={source} offset={offset} batch_size={window_batch_size} seq_len={window_seq_len} tokens={} logits={} loss={:.6} loss_ema={:.6} delta={} finite={} nonzero={} adam_lr={:.6e} aurora_lr={:.6e} forward_ms={:.3} backward_enqueue_ms={:.3} loss_sync_ms={:.3} optimizer_ms={:.3} aurora_ms={:.3} adam_ms={:.3} embed_lookup_ms={:.3} token_embed_ms={:.3} final_norm_ms={:.3} blocks_ms={:.3}",
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
            previous_loss = Some(stats.loss);
        }
        if should_eval_step(step, steps, eval_interval) {
            let val_loss = trainer.eval_loss(&validation_batch)?;
            println!("eval step={step} val_loss={val_loss:.6}");
        }
        if let Some(trace) = &stats.diagnostics {
            println!(
                "trace step={step} updates={} positive_update_dot={} zero_grad_changed={} max_update_to_weight_rms={:.6e} dlogits_rms={:.6e} dlogits_max={:.6e} d_lm_head_rms={:.6e} d_lm_head_max={:.6e} d_embedding_rms={:.6e} d_embedding_max={:.6e} token_embedding_global_before={:.6e} token_embedding_global_after={:.6e} token_embedding_changed_bytes={} token_embedding_hash_before={:016x} token_embedding_hash_after={:016x}",
                trace.update_count,
                trace.positive_update_dot_count,
                trace.zero_grad_changed_count,
                trace.max_update_to_weight_rms,
                trace.dlogits_rms,
                trace.dlogits_max,
                trace.d_lm_head_rms,
                trace.d_lm_head_max,
                trace.d_embedding_rms,
                trace.d_embedding_max,
                trace.token_embedding_global_before,
                trace.token_embedding_global_after,
                trace.token_embedding_changed_bytes,
                trace.token_embedding_hash_before,
                trace.token_embedding_hash_after,
            );
            for update in &trace.updates {
                println!(
                    "update step={step} tensor={} len={} grad_rms={:.6e} grad_max={:.6e} grad_nonzero={} grad_finite={} weight_rms_before={:.6e} weight_rms_after={:.6e} delta_rms={:.6e} delta_max={:.6e} update_to_weight_rms={:.6e} delta_grad_dot={:.6e} delta_grad_cos={:.6e} predicted_delta_rms={:.6e} predicted_delta_grad_dot={:.6e} predicted_delta_grad_cos={:.6e} quant_error_rms={:.6e} quant_error_to_predicted_delta_rms={:.6e} changed_bytes={} changed_scales={} global_before={:.6e} global_after={:.6e}",
                    update.name,
                    update.len,
                    update.grad_rms,
                    update.grad_max,
                    update.grad_nonzero,
                    update.grad_finite,
                    update.weight_rms_before,
                    update.weight_rms_after,
                    update.delta_rms,
                    update.delta_max,
                    update.update_to_weight_rms,
                    update.delta_grad_dot,
                    update.delta_grad_cos,
                    update.predicted_delta_rms,
                    update.predicted_delta_grad_dot,
                    update.predicted_delta_grad_cos,
                    update.quant_error_rms,
                    update.quant_error_to_predicted_delta_rms,
                    update.changed_bytes,
                    update.changed_scales,
                    update.global_before,
                    update.global_after,
                );
            }
        }
    }

    if let Some(path) = train_save_model_path() {
        trainer.save_model(&path)?;
        println!("saved_model={}", path.display());
    }

    if let Some(path) = train_loss_graph_path() {
        let path = loss_curve.write_png(&path)?;
        println!("loss_graph={}", path.display());
    }

    if let Some(prompt) = train_generate_prompt() {
        let text = trainer.generate_greedy(&prompt, train_generate_tokens())?;
        println!("generated_text_begin");
        println!("{text}");
        println!("generated_text_end");
    }

    Ok(())
}

fn train_steps() -> usize {
    std::env::var("TRAIN_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_TRAIN_STEPS)
}

fn train_log_interval() -> usize {
    std::env::var("TRAIN_LOG_INTERVAL")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1)
        .max(1)
}

fn train_eval_interval() -> Option<usize> {
    std::env::var("TRAIN_EVAL_INTERVAL")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|interval| *interval > 0)
}

fn train_save_model_path() -> Option<PathBuf> {
    std::env::var("TRAIN_SAVE_MODEL")
        .ok()
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn train_loss_graph_path() -> Option<PathBuf> {
    std::env::var("TRAIN_LOSS_GRAPH")
        .ok()
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn train_generate_prompt() -> Option<String> {
    std::env::var("TRAIN_GENERATE_PROMPT")
        .ok()
        .filter(|value| !value.is_empty())
}

fn train_generate_tokens() -> usize {
    std::env::var("TRAIN_GENERATE_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(128)
}

fn should_log_step(step: usize, steps: usize, log_interval: usize) -> bool {
    step == 0 || step + 1 == steps || step % log_interval == 0
}

fn should_eval_step(step: usize, steps: usize, eval_interval: Option<usize>) -> bool {
    eval_interval.is_some_and(|interval| step == 0 || step + 1 == steps || step % interval == 0)
}

fn update_loss_ema(loss_ema: &mut Option<f32>, loss: f32) -> f32 {
    const BETA: f32 = 0.9;
    let next = loss_ema
        .map(|ema| BETA * ema + (1.0 - BETA) * loss)
        .unwrap_or(loss);
    *loss_ema = Some(next);
    next
}
