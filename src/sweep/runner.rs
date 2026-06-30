use std::{fs, path::PathBuf};

use super::{
    analysis, baseline::Baseline, chain, config::SweepConfig, fmt, history::History, optimizer,
    proposal_log,
};
use crate::time_utils;

mod execute;
mod trial_record;

use execute::{run_trial, screen_baseline};
use trial_record::{current_baseline_trial, promoted_screen_loss};

pub fn run(config: SweepConfig) -> Result<(), Box<dyn std::error::Error>> {
    let sweep_dir = config.sweep_dir.clone().unwrap_or_else(default_sweep_dir);
    fs::create_dir_all(&sweep_dir)?;
    let mut history = History::load(sweep_dir.join("trials.tsv"))?;
    let mut shared_history = History::load(config.seed_history.clone())?;
    chain::sync_shared_history(&mut shared_history, &history.trials, config.dry_run)?;
    let mut baseline = Baseline::load(config.baseline.clone())?;
    let mut baseline_screen_trial = screen_baseline(&baseline, &config, &sweep_dir)?;
    let mut baseline_screen_loss = baseline_screen_trial
        .as_ref()
        .and_then(|trial| trial.screen_val_loss);
    let mut rng = chain::sweep_rng(config.seed, history.trials.len());

    for index in history.trials.len()..config.trials {
        let baseline_trial =
            current_baseline_trial(baseline_screen_trial.as_ref(), baseline.measured_trial());
        let all_trials = chain::all_trials_with_baseline(
            baseline_trial.as_ref(),
            &shared_history.trials,
            &history.trials,
        );
        let sweep_analysis = analysis::analyze(&all_trials, &config);
        analysis::write(&sweep_dir, &sweep_analysis, &config)?;
        analysis::print_summary(&sweep_analysis);
        let seen = chain::seen_keys(&all_trials);
        let proposal = optimizer::propose(
            &all_trials,
            &seen,
            &mut rng,
            &config,
            &sweep_analysis,
            baseline.candidate(),
        );
        proposal_log::write(&sweep_dir, index, &proposal)?;
        let screen_score = proposal
            .selected_scored()
            .map(|scored| scored.score.clone());
        let candidate = proposal.candidate;
        let trial_dir = sweep_dir.join(format!("trial_{index:04}"));
        println!("sweep_trial_begin index={index} key={}", candidate.key());
        let trial = run_trial(
            index,
            &sweep_dir,
            &trial_dir,
            candidate,
            &config,
            baseline_screen_loss,
            screen_score.as_ref(),
        )?;
        println!(
            "sweep_trial_end index={index} status={} val_loss={} completed_steps={} log_path={}",
            trial.status,
            trial
                .val_loss
                .map(fmt::f64_6)
                .unwrap_or_else(|| "NaN".to_string()),
            fmt::optional_usize(trial.completed_steps),
            trial.log_path.display()
        );
        history.append_unique(trial.clone())?;
        let promoted = baseline.promote_trial(&trial, config.dry_run)?;
        if promoted {
            if let Some(loss) = promoted_screen_loss(&trial) {
                baseline_screen_loss = Some(loss);
                baseline_screen_trial = Some(trial.clone());
            } else {
                baseline_screen_trial = screen_baseline(&baseline, &config, &sweep_dir)?;
                baseline_screen_loss = baseline_screen_trial
                    .as_ref()
                    .and_then(|trial| trial.screen_val_loss);
            }
            println!(
                "sweep_baseline_promoted val_loss={:.6} key={} path={}",
                baseline.val_loss().unwrap_or(f64::NAN),
                baseline
                    .candidate()
                    .map(|candidate| candidate.key())
                    .unwrap_or_default(),
                config.baseline.display()
            );
        }
        if !config.dry_run {
            shared_history.append_unique(trial)?;
        }
        let baseline_trial =
            current_baseline_trial(baseline_screen_trial.as_ref(), baseline.measured_trial());
        let all_trials = chain::all_trials_with_baseline(
            baseline_trial.as_ref(),
            &shared_history.trials,
            &history.trials,
        );
        let sweep_analysis = analysis::analyze(&all_trials, &config);
        analysis::write(&sweep_dir, &sweep_analysis, &config)?;
    }
    Ok(())
}

fn default_sweep_dir() -> PathBuf {
    PathBuf::from("target/sweeps").join(time_utils::utc_compact_stamp())
}
