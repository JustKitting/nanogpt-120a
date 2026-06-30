use std::path::PathBuf;

use crate::sweep::{candidate::Candidate, history::Trial};

pub(in crate::sweep) fn trial(status: &str, val_loss: Option<f64>, candidate: Candidate) -> Trial {
    Trial {
        candidate,
        status: status.to_string(),
        val_loss,
        completed_steps: Some(10),
        elapsed_s: Some(900.0),
        screen_val_loss: val_loss.map(|loss| loss + 1.0),
        screen_completed_steps: Some(10),
        screen_elapsed_s: Some(30.0),
        screen_reason: Some("screen_loss_improved".to_string()),
        log_path: PathBuf::from("train.log"),
    }
}

pub(in crate::sweep) fn success_trial(candidate: Candidate, val_loss: f64) -> Trial {
    trial("success", Some(val_loss), candidate)
}

pub(in crate::sweep) fn trial_with_losses(
    candidate: Candidate,
    val_loss: f64,
    screen_loss: f64,
) -> Trial {
    Trial {
        screen_val_loss: Some(screen_loss),
        ..success_trial(candidate, val_loss)
    }
}

pub(in crate::sweep) fn trial_with_status(candidate: Candidate, status: &str) -> Trial {
    Trial {
        screen_val_loss: None,
        screen_completed_steps: None,
        screen_elapsed_s: None,
        screen_reason: None,
        ..trial(status, None, candidate)
    }
}
