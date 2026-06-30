use std::fs::File;
use std::io::{BufWriter, Write};

use super::super::super::diagnostics::{TensorUpdateDiagnostics, TrainingDiagnostics};
use super::super::super::launch::CudaTrainOutput;
use crate::AppResult;

pub(super) const SUMMARY_HEADER: &str = "step\tsource\twindow_offset\tbatch_size\tseq_len\tupdate_count\tpositive_update_dot_count\tzero_grad_changed_count\tmax_update_to_weight_rms\tdlogits_rms\tdlogits_max\td_lm_head_rms\td_lm_head_max\td_embedding_rms\td_embedding_max\ttoken_embedding_global_before\ttoken_embedding_global_after\ttoken_embedding_changed_bytes\ttoken_embedding_hash_before\ttoken_embedding_hash_after";
pub(super) const TENSOR_HEADER: &str = "step\tsource\twindow_offset\tbatch_size\tseq_len\ttensor\tlen\tgrad_rms\tgrad_max\tgrad_nonzero\tgrad_finite\tweight_rms_before\tweight_rms_after\tdelta_rms\tdelta_max\tupdate_to_weight_rms\tdelta_grad_dot\tdelta_grad_cos\tpredicted_delta_rms\tpredicted_delta_grad_dot\tpredicted_delta_grad_cos\tquant_error_rms\tquant_error_to_predicted_delta_rms\tchanged_bytes\tchanged_scales\tglobal_before\tglobal_after";

pub(super) fn write_summary(
    writer: &mut BufWriter<File>,
    step: usize,
    item: &CudaTrainOutput,
    trace: &TrainingDiagnostics,
) -> AppResult {
    writeln!(
        writer,
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
    Ok(())
}

pub(super) fn write_tensor(
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
