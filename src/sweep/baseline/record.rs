use std::path::PathBuf;

use crate::sweep::candidate::Candidate;

#[derive(Clone)]
pub(super) struct Record {
    pub(super) val_loss: f64,
    pub(super) completed_steps: Option<usize>,
    pub(super) elapsed_s: Option<f64>,
    pub(super) screen_loss: Option<f64>,
    pub(super) screen_completed_steps: Option<usize>,
    pub(super) screen_elapsed_s: Option<f64>,
    pub(super) screen_reason: Option<String>,
    pub(super) log_path: PathBuf,
    pub(super) candidate: Candidate,
}
