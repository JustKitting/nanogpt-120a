use std::fs;
use std::path::{Path, PathBuf};

use crate::sweep::candidate::Candidate;
use crate::sweep::history::Trial;

pub(super) fn current_baseline_trial(
    screen_trial: Option<&Trial>,
    measured_trial: Option<Trial>,
) -> Option<Trial> {
    screen_trial.cloned().or(measured_trial)
}

pub(super) fn trial(
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

pub(super) fn trial_with_log(
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

pub(super) fn promoted_screen_loss(trial: &Trial) -> Option<f64> {
    let text = fs::read_to_string(trial.log_path.with_file_name("screen_decision.env")).ok()?;
    value(&text, "SCREEN_LOSS")?.parse().ok()
}

fn value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines().find_map(|line| {
        let (name, value) = line.split_once('=')?;
        (name == key).then_some(value)
    })
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::{current_baseline_trial, promoted_screen_loss};
    use crate::sweep::candidate::Candidate;
    use crate::sweep::history::Trial;

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

    #[test]
    fn current_screened_baseline_replaces_stale_baseline_screen_metrics() {
        let measured = Trial {
            candidate: candidate(),
            status: "success".to_string(),
            val_loss: Some(3.5),
            completed_steps: Some(3000),
            elapsed_s: Some(900.0),
            screen_val_loss: Some(6.9),
            screen_completed_steps: Some(500),
            screen_elapsed_s: Some(180.0),
            screen_reason: Some("old_screen".to_string()),
            log_path: PathBuf::from("old/train.log"),
        };
        let screened = Trial {
            screen_val_loss: Some(6.2),
            screen_completed_steps: Some(100),
            screen_elapsed_s: Some(30.0),
            screen_reason: Some("screen_baseline".to_string()),
            log_path: PathBuf::from("screen_baseline/screen.log"),
            ..measured.clone()
        };

        let current = current_baseline_trial(Some(&screened), Some(measured)).unwrap();

        assert_eq!(current.screen_val_loss, Some(6.2));
        assert_eq!(current.screen_elapsed_s, Some(30.0));
        assert_eq!(
            current.log_path,
            PathBuf::from("screen_baseline/screen.log")
        );
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
            nextlat_lr_scale: 1.0,
            warmup_steps: 20,
            start_ratio: 0.1,
            amuse_beta1: 0.4,
            amuse_rho: 0.8,
        }
    }
}
