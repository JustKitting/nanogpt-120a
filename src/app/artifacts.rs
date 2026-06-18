use std::path::PathBuf;

use crate::{AppResult, app::config};

use super::loss_graph::LossCurve;
use super::run_output::{self, RunOutput};

pub fn write_loss_graph(run_output: &RunOutput, loss_curve: &LossCurve) -> AppResult<PathBuf> {
    let path = config::loss_graph_path(run_output);
    run_output::ensure_parent(&path)?;
    loss_curve.write_png(&path)
}

pub fn write_generated_text(run_output: &RunOutput, text: &str) -> AppResult<PathBuf> {
    let path = run_output.path("generated.txt");
    run_output::ensure_parent(&path)?;
    std::fs::write(&path, text)?;
    Ok(path)
}
