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

pub fn run(config: SweepConfig) -> Result<(), Box<dyn std::error::Error>> {
    let sweep_dir = config.sweep_dir.clone().unwrap_or_else(default_sweep_dir);
    fs::create_dir_all(&sweep_dir)?;
    let mut history = History::load(sweep_dir.join("trials.tsv"))?;
    let mut shared_history = History::load(config.seed_history.clone())?;
    chain::sync_shared_history(&mut shared_history, &history.trials, config.dry_run)?;
    let mut baseline = Baseline::load(config.baseline.clone())?;
    let initial_trials = chain::all_trials(&shared_history.trials, &history.trials);
    if baseline.promote_best(&initial_trials, config.dry_run)? {
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
    let mut baseline_screen_loss = screen_baseline(&baseline, &config, &sweep_dir)?;
    let mut rng = chain::sweep_rng(config.seed, history.trials.len());

    for index in history.trials.len()..config.trials {
        let baseline_trial = baseline.measured_trial();
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
            baseline_screen_loss = if let Some(loss) = promoted_screen_loss(&trial) {
                Some(loss)
            } else {
                screen_baseline(&baseline, &config, &sweep_dir)?
            };
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
        let baseline_trial = baseline.measured_trial();
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
) -> Result<Option<f64>, Box<dyn std::error::Error>> {
    let Some(candidate) = baseline.candidate().cloned() else {
        return Ok(None);
    };
    if let Some(loss) = baseline.screen_loss() {
        println!("sweep_screen_baseline_cached val_loss={loss:.6}");
        return Ok(Some(loss));
    }
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
    if result.completed_steps.unwrap_or(0) < config.screen_steps {
        println!("sweep_screen_baseline_failed=incomplete");
        return Ok(None);
    }

    if let Some(val_loss) = result.val_loss {
        println!(
            "sweep_screen_baseline val_loss={val_loss:.6} completed_steps={}",
            result
                .completed_steps
                .map(|value| value.to_string())
                .unwrap_or_default()
        );
        Ok(Some(val_loss))
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
    let screen_decision = screen_gate::decide(
        &screen_result,
        screen_baseline,
        config.screen_steps,
        screen_score,
    );
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

fn trial(
    candidate: Candidate,
    status: &str,
    val_loss: Option<f64>,
    completed_steps: Option<usize>,
    elapsed_s: Option<f64>,
    screen_val_loss: Option<f64>,
    screen_completed_steps: Option<usize>,
    screen_elapsed_s: Option<f64>,
    screen_reason: Option<&str>,
    trial_dir: &Path,
) -> Trial {
    trial_with_log(
        candidate,
        status,
        val_loss,
        completed_steps,
        elapsed_s,
        screen_val_loss,
        screen_completed_steps,
        screen_elapsed_s,
        screen_reason,
        trial_dir,
        "train.log",
    )
}

fn promoted_screen_loss(trial: &Trial) -> Option<f64> {
    let text = fs::read_to_string(trial.log_path.with_file_name("screen_decision.env")).ok()?;
    value(&text, "SCREEN_LOSS")?.parse().ok()
}

fn value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines().find_map(|line| {
        let (name, value) = line.split_once('=')?;
        (name == key).then_some(value)
    })
}

fn selected_score(proposal: &optimizer::Proposal) -> Option<analysis::CandidateScore> {
    proposal
        .ranked
        .iter()
        .find(|scored| scored.candidate.key() == proposal.candidate.key())
        .map(|scored| scored.score.clone())
}

fn trial_with_log(
    candidate: Candidate,
    status: &str,
    val_loss: Option<f64>,
    completed_steps: Option<usize>,
    elapsed_s: Option<f64>,
    screen_val_loss: Option<f64>,
    screen_completed_steps: Option<usize>,
    screen_elapsed_s: Option<f64>,
    screen_reason: Option<&str>,
    trial_dir: &Path,
    log_name: &str,
) -> Trial {
    Trial {
        candidate,
        status: status.to_string(),
        val_loss,
        completed_steps,
        log_path: PathBuf::from(trial_dir).join(log_name),
        elapsed_s,
        screen_val_loss,
        screen_completed_steps,
        screen_elapsed_s,
        screen_reason: screen_reason.map(ToString::to_string),
    }
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

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{Trial, promoted_screen_loss};
    use crate::sweep::candidate::Candidate;

    #[test]
    fn reads_promoted_screen_loss_from_decision_artifact() {
        let dir = std::env::temp_dir().join(format!("sweep-screen-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("screen_decision.env"), "SCREEN_LOSS=3.250000\n").unwrap();
        let trial = Trial {
            candidate: candidate(),
            status: "success".to_string(),
            val_loss: Some(3.0),
            completed_steps: Some(100),
            elapsed_s: Some(900.0),
            screen_val_loss: Some(3.25),
            screen_completed_steps: Some(500),
            screen_elapsed_s: Some(90.0),
            screen_reason: Some("screen_loss_improved".to_string()),
            log_path: dir.join("train.log"),
        };

        assert_eq!(promoted_screen_loss(&trial), Some(3.25));
        let _ = fs::remove_dir_all(dir);
    }

    fn candidate() -> Candidate {
        Candidate {
            batch_size: 8,
            n_layer: 4,
            n_embd: 1024,
            n_head: 16,
            aurora_phases: 4,
            aurora_blocks: 80,
            lr_scale: 1.0,
            adam_lr_scale: 1.0,
            warmup_steps: 20,
            start_ratio: 0.1,
            amuse_beta1: 0.4,
            amuse_rho: 0.8,
        }
    }
}
