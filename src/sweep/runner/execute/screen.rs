use std::{fs, path::Path};

use crate::sweep::{baseline::Baseline, config::SweepConfig, history::Trial, run_build, run_train};
use crate::sweep::SweepResult;

pub(in crate::sweep::runner) fn screen_baseline(
    baseline: &Baseline,
    config: &SweepConfig,
    sweep_dir: &Path,
) -> SweepResult<Option<Trial>> {
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
