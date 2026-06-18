mod app;
mod checkpoint;
mod loss_graph;
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
    let validation_tokens = data.validation_tokens()?;
    let validation_batch = trainer.batch_from_default_windows(&validation_tokens)?;

    run_output.write_info(&app::run_info::build(&dataset, &config))?;
    println!(
        "training_tokens={} steps={}",
        data.token_count(),
        config.steps
    );

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
