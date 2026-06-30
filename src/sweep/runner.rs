use std::{
    fs,
    path::{Path, PathBuf},
};

use time::OffsetDateTime;

use super::{
    analysis,
    baseline::Baseline,
    candidate::Candidate,
    chain,
    config::SweepConfig,
    history::{self, History, Trial},
    optimizer,
    parse::RunResult,
    proposal_log, run_build, run_train, screen_gate, status,
};

mod trial_record;

use trial_record::{current_baseline_trial, promoted_screen_loss, trial, trial_with_log};

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
        let screen_score = selected_score(&proposal);
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
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "NaN".to_string()),
            trial
                .completed_steps
                .map(|value| value.to_string())
                .unwrap_or_default(),
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

fn screen_baseline(
    baseline: &Baseline,
    config: &SweepConfig,
    sweep_dir: &Path,
) -> Result<Option<Trial>, Box<dyn std::error::Error>> {
    let Some(mut trial) = baseline.measured_trial() else {
        return Ok(None);
    };
    let candidate = trial.candidate.clone();
    if config.dry_run {
        return Ok(None);
    }

    let trial_dir = sweep_dir.join("screen_baseline");
    fs::create_dir_all(&trial_dir)?;
    let build_status =
        run_build::build_candidate(&candidate, config, &trial_dir.join("build.log"))?;
    if !build_status.success() {
        println!("sweep_screen_baseline_failed=build");
        return Ok(None);
    }

    let result = run_train::run_screen_candidate(&candidate, config, sweep_dir, &trial_dir, 0)?;
    if let Some(val_loss) = result.val_loss {
        println!(
            "sweep_screen_baseline val_loss={val_loss:.6} completed_steps={}",
            result
                .completed_steps
                .map(|value| value.to_string())
                .unwrap_or_default()
        );
        trial.screen_val_loss = Some(val_loss);
        trial.screen_completed_steps = result.completed_steps;
        trial.screen_elapsed_s = result.last_elapsed_s;
        trial.screen_reason = Some("screen_baseline".to_string());
        trial.log_path = trial_dir.join("screen.log");
        Ok(Some(trial))
    } else {
        println!("sweep_screen_baseline_failed=run");
        Ok(None)
    }
}

fn run_trial(
    index: usize,
    sweep_dir: &Path,
    trial_dir: &Path,
    candidate: Candidate,
    config: &SweepConfig,
    screen_baseline: Option<f64>,
    screen_score: Option<&analysis::CandidateScore>,
) -> Result<Trial, Box<dyn std::error::Error>> {
    fs::create_dir_all(trial_dir)?;
    history::write_candidate(&trial_dir.join("candidate.env"), &candidate)?;
    let mut run_result = RunResult::default();
    status::record(
        sweep_dir,
        trial_dir,
        index,
        &candidate,
        "trial_started",
        &run_result,
    )?;
    if config.dry_run {
        status::record(
            sweep_dir,
            trial_dir,
            index,
            &candidate,
            "dry_run",
            &run_result,
        )?;
        return Ok(trial(
            candidate, "dry_run", None, None, None, None, None, None, None, trial_dir,
        ));
    }

    status::record(
        sweep_dir,
        trial_dir,
        index,
        &candidate,
        "build_started",
        &run_result,
    )?;
    let build_status =
        run_build::build_candidate(&candidate, config, &trial_dir.join("build.log"))?;
    if !build_status.success() {
        status::record(
            sweep_dir,
            trial_dir,
            index,
            &candidate,
            "failed_build",
            &run_result,
        )?;
        return Ok(trial(
            candidate,
            "failed_build",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            trial_dir,
        ));
    }

    let screen_result =
        run_train::run_screen_candidate(&candidate, config, sweep_dir, trial_dir, index)?;
    let screen_decision = screen_gate::decide(&screen_result, screen_baseline, screen_score);
    screen_gate::write(&trial_dir.join("screen_decision.env"), &screen_decision)?;
    if !screen_decision.pass {
        status::record(
            sweep_dir,
            trial_dir,
            index,
            &candidate,
            &format!("rejected_screen_{}", screen_decision.reason),
            &screen_result,
        )?;
        return Ok(trial_with_log(
            candidate,
            "rejected_screen",
            None,
            screen_result.completed_steps,
            screen_result.last_elapsed_s,
            screen_result.val_loss,
            screen_result.completed_steps,
            screen_result.last_elapsed_s,
            Some(screen_decision.reason),
            trial_dir,
            "screen.log",
        ));
    }
    status::record(
        sweep_dir,
        trial_dir,
        index,
        &candidate,
        "screen_passed",
        &screen_result,
    )?;

    run_result = run_train::run_candidate(&candidate, config, sweep_dir, trial_dir, index)?;
    let status_name = match (run_result.val_loss, run_result.saw_nan) {
        (Some(_), false) => "success",
        (Some(_), true) => "nan_with_val",
        (None, true) => "nan",
        (None, false) => "failed_run",
    };
    status::record(
        sweep_dir,
        trial_dir,
        index,
        &candidate,
        status_name,
        &run_result,
    )?;
    Ok(trial(
        candidate,
        status_name,
        run_result.val_loss,
        run_result.completed_steps,
        run_result.last_elapsed_s,
        screen_result.val_loss,
        screen_result.completed_steps,
        screen_result.last_elapsed_s,
        Some(screen_decision.reason),
        trial_dir,
    ))
}

fn selected_score(proposal: &optimizer::Proposal) -> Option<analysis::CandidateScore> {
    proposal
        .ranked
        .iter()
        .find(|scored| scored.candidate.key() == proposal.candidate.key())
        .map(|scored| scored.score.clone())
}

fn default_sweep_dir() -> PathBuf {
    PathBuf::from("target/sweeps").join(utc_stamp())
}

fn utc_stamp() -> String {
    let now = OffsetDateTime::now_utc();
    format!(
        "{:04}{:02}{:02}_{:02}{:02}{:02}Z",
        now.year(),
        now.month() as u8,
        now.day(),
        now.hour(),
        now.minute(),
        now.second()
    )
}
