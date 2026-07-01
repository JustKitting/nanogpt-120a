use std::{fs, io, path::Path};

use super::{super::config::SweepConfig, SweepAnalysis, beliefs};

mod summary;
mod tsv;

pub fn write(sweep_dir: &Path, analysis: &SweepAnalysis, config: &SweepConfig) -> io::Result<()> {
    for (file_name, text) in [
        ("analysis_summary.md", summary::text(analysis)),
        ("analysis_effects.tsv", tsv::effects(analysis)),
        ("analysis_interactions.tsv", tsv::interactions(analysis)),
        ("analysis_beliefs.tsv", beliefs::tsv(analysis, config)),
    ] {
        fs::write(sweep_dir.join(file_name), text)?;
    }
    Ok(())
}
