use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

mod rows;

use super::super::diagnostics;
use super::super::launch::CudaTrainOutput;
use crate::AppResult;

pub(in crate::training) struct DebugTraceLogger {
    summary: Option<BufWriter<File>>,
    tensors: Option<BufWriter<File>>,
}

impl DebugTraceLogger {
    pub(in crate::training) fn new(directory: PathBuf) -> AppResult<Self> {
        if !diagnostics::enabled() {
            return Ok(Self {
                summary: None,
                tensors: None,
            });
        }

        fs::create_dir_all(&directory)?;
        let mut summary = BufWriter::new(File::create(directory.join("optimizer_summary.tsv"))?);
        let mut tensors = BufWriter::new(File::create(directory.join("optimizer_tensors.tsv"))?);
        writeln!(summary, "{}", rows::SUMMARY_HEADER)?;
        writeln!(tensors, "{}", rows::TENSOR_HEADER)?;
        println!("debug_metrics_dir={}", directory.display());
        Ok(Self {
            summary: Some(summary),
            tensors: Some(tensors),
        })
    }

    pub(in crate::training) fn log_train_step(
        &mut self,
        step: usize,
        item: &CudaTrainOutput,
    ) -> AppResult {
        let Some(trace) = item.stats.diagnostics.as_ref() else {
            return Ok(());
        };

        if let Some(summary) = self.summary.as_mut() {
            rows::write_summary(summary, step, item, trace)?;
        }

        if let Some(tensors) = self.tensors.as_mut() {
            for update in &trace.updates {
                rows::write_tensor(tensors, step, item, update)?;
            }
        }

        Ok(())
    }
}
