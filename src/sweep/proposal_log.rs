use std::{fs, path::Path};

use super::{optimizer::Proposal, SweepResult};

mod format;

use format::{ranked_tsv, selected_text, sources_tsv};

pub fn write(sweep_dir: &Path, index: usize, proposal: &Proposal) -> SweepResult {
    fs::write(
        sweep_dir.join(format!("candidate_{index:04}_score.txt")),
        selected_text(proposal),
    )?;
    fs::write(
        sweep_dir.join(format!("candidate_{index:04}_ranked.tsv")),
        ranked_tsv(proposal),
    )?;
    fs::write(
        sweep_dir.join(format!("candidate_{index:04}_sources.tsv")),
        sources_tsv(proposal),
    )?;
    Ok(())
}
