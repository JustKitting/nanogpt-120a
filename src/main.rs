mod app;
mod checkpoint;
mod training;
mod upload;

use std::error::Error;

use app::config::TrainConfig;
use app::logging::{StepLogContext, TrainingLogger};
use training::{TokenDataLoader, Trainer};

type AppResult<T = ()> = Result<T, Box<dyn Error>>;

fn main() -> AppResult {
    let mut trainer = Trainer::new(app::config::SEED)?;
    let dataset = TokenDataLoader::training_dataset_name();
    let config = TrainConfig::from_env();
    let run_output = app::run_output::RunOutput::new(&dataset, config.steps)?;
    println!("run_dir={}", run_output.dir().display());

    if let Some(path) = app::config::load_model_path() {
        trainer.load_model(&path)?;
        println!("loaded_model={}", path.display());
    }

    let mut data = TokenDataLoader::from_training_dataset()?;
    let mut logger = TrainingLogger::new();
    let wall_clock = app::wall_clock::WallClockBudget::new(config.max_seconds);
    let validation_tokens = data.validation_tokens()?;
    let validation_batch = trainer.batch_from_default_windows(&validation_tokens)?;

    run_output.write_info(&app::run_info::build(&dataset, &config))?;
    println!(
        "training_tokens={} steps={}",
        data.token_count(),
        config.steps
    );

    let mut completed_steps = 0usize;
    for step in 0..config.steps {
        let log_step = app::config::should_log_step(step, config.steps, config.log_interval);
        let window = data.next_batch()?;
        let source = window.source.display().to_string();
        let batch = trainer.batch_from_default_windows(&window.tokens)?;
        let stats = trainer.train_step(&batch, log_step)?;

        if log_step {
            logger.log_step(
                StepLogContext {
                    step,
                    elapsed_s: wall_clock.elapsed_seconds(),
                    source: &source,
                    offset: window.offset,
                    batch_size: window.batch_size,
                    seq_len: window.seq_len,
                },
                &stats,
            );
        }
        if app::config::should_eval_step(step, config.steps, config.eval_interval) {
            let val_loss = trainer.eval_loss(&validation_batch)?;
            println!("eval step={step} val_loss={val_loss:.6}");
        }
        app::logging::log_diagnostics(step, &stats);
        completed_steps = step + 1;
        if wall_clock.expired() {
            println!(
                "stopped_by_wall_clock=true elapsed_s={:.3} completed_steps={}",
                wall_clock.elapsed_seconds(),
                step + 1,
            );
            break;
        }
    }
    let train_elapsed_s = wall_clock.elapsed_seconds();

    if config.max_seconds.is_some() {
        let eval_start = std::time::Instant::now();
        let val_loss = trainer.eval_loss(&validation_batch)?;
        let eval_elapsed_s = eval_start.elapsed().as_secs_f64();
        println!(
            "heldout_eval split=val val_loss={val_loss:.6} train_elapsed_s={train_elapsed_s:.3} eval_elapsed_s={eval_elapsed_s:.3} completed_steps={completed_steps}",
        );
    }

    if let Some(path) = app::config::save_model_path(&run_output) {
        app::run_output::ensure_parent(&path)?;
        trainer.save_model(&path)?;
        println!("saved_model={}", path.display());
    }

    let loss_graph_path = app::artifacts::write_loss_graph(&run_output, logger.loss_curve())?;
    println!("loss_graph={}", loss_graph_path.display());

    if let Some(prompt) = app::config::generate_prompt() {
        let text = trainer.generate_sampled(
            &prompt,
            app::config::generate_tokens(),
            app::config::sampling_config(),
        )?;
        let generated_path = app::artifacts::write_generated_text(&run_output, &text)?;
        println!("generated_text={}", generated_path.display());
        println!("generated_text_begin");
        println!("{text}");
        println!("generated_text_end");
    }

    Ok(())
}
