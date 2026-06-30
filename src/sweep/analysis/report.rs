use std::{fs, io, path::Path};

use super::{super::config::SweepConfig, SweepAnalysis, beliefs};

mod summary;
mod tsv;

pub fn write(sweep_dir: &Path, analysis: &SweepAnalysis, config: &SweepConfig) -> io::Result<()> {
    fs::write(
        sweep_dir.join("analysis_summary.md"),
        summary::text(analysis),
    )?;
    fs::write(
        sweep_dir.join("analysis_effects.tsv"),
        tsv::effects(analysis),
    )?;
    fs::write(
        sweep_dir.join("analysis_interactions.tsv"),
        tsv::interactions(analysis),
    )?;
    fs::write(
        sweep_dir.join("analysis_beliefs.tsv"),
        beliefs::tsv(analysis, config),
    )
}
