use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use super::super::diagnostics::{self, TensorUpdateDiagnostics};
use super::super::launch::CudaTrainOutput;
use crate::AppResult;

const SUMMARY_HEADER: &str = "step\tsource\twindow_offset\tbatch_size\tseq_len\tupdate_count\tpositive_update_dot_count\tzero_grad_changed_count\tmax_update_to_weight_rms\tdlogits_rms\tdlogits_max\td_lm_head_rms\td_lm_head_max\td_embedding_rms\td_embedding_max\ttoken_embedding_global_before\ttoken_embedding_global_after\ttoken_embedding_changed_bytes\ttoken_embedding_hash_before\ttoken_embedding_hash_after";
const TENSOR_HEADER: &str = "step\tsource\twindow_offset\tbatch_size\tseq_len\ttensor\tlen\tgrad_rms\tgrad_max\tgrad_nonzero\tgrad_finite\tweight_rms_before\tweight_rms_after\tdelta_rms\tdelta_max\tupdate_to_weight_rms\tdelta_grad_dot\tdelta_grad_cos\tpredicted_delta_rms\tpredicted_delta_grad_dot\tpredicted_delta_grad_cos\tquant_error_rms\tquant_error_to_predicted_delta_rms\tchanged_bytes\tchanged_scales\tglobal_before\tglobal_after";

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
        writeln!(summary, "{SUMMARY_HEADER}")?;
        writeln!(tensors, "{TENSOR_HEADER}")?;
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
            writeln!(
                summary,
                "{step}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:016x}\t{:016x}",
                tsv_field(&item.source),
                item.window_offset,
                item.batch_size,
                item.seq_len,
                trace.update_count,
                trace.positive_update_dot_count,
                trace.zero_grad_changed_count,
                trace.max_update_to_weight_rms,
                trace.dlogits_rms,
                trace.dlogits_max,
                trace.d_lm_head_rms,
                trace.d_lm_head_max,
                trace.d_embedding_rms,
                trace.d_embedding_max,
                trace.token_embedding_global_before,
                trace.token_embedding_global_after,
                trace.token_embedding_changed_bytes,
                trace.token_embedding_hash_before,
                trace.token_embedding_hash_after,
            )?;
        }

        if let Some(tensors) = self.tensors.as_mut() {
            for update in &trace.updates {
                write_tensor_row(tensors, step, item, update)?;
            }
        }

        Ok(())
    }
}

fn write_tensor_row(
    writer: &mut BufWriter<File>,
    step: usize,
    item: &CudaTrainOutput,
    update: &TensorUpdateDiagnostics,
) -> AppResult {
    writeln!(
        writer,
        "{step}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        tsv_field(&item.source),
        item.window_offset,
        item.batch_size,
        item.seq_len,
        tsv_field(&update.name),
        update.len,
        update.grad_rms,
        update.grad_max,
        update.grad_nonzero,
        update.grad_finite,
        update.weight_rms_before,
        update.weight_rms_after,
        update.delta_rms,
        update.delta_max,
        update.update_to_weight_rms,
        update.delta_grad_dot,
        update.delta_grad_cos,
        update.predicted_delta_rms,
        update.predicted_delta_grad_dot,
        update.predicted_delta_grad_cos,
        update.quant_error_rms,
        update.quant_error_to_predicted_delta_rms,
        update.changed_bytes,
        update.changed_scales,
        update.global_before,
        update.global_after,
    )?;
    Ok(())
}

fn tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}
