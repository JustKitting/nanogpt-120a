use std::{fs, path::Path};

use super::{SweepResult, optimizer::Proposal};

mod format;

use format::{ranked_tsv, selected_text, sources_tsv};

pub fn write(sweep_dir: &Path, index: usize, proposal: &Proposal) -> SweepResult {
    for (suffix, text) in [
        ("score.txt", selected_text(proposal)),
        ("ranked.tsv", ranked_tsv(proposal)),
        ("sources.tsv", sources_tsv(proposal)),
    ] {
        fs::write(
            sweep_dir.join(format!("candidate_{index:04}_{suffix}")),
            text,
        )?;
    }
    Ok(())
}
