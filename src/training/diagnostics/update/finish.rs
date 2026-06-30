use cuda_core::CudaStream;

use crate::AppResult;
use crate::upload::{UploadedLayerNorm, UploadedLinear, UploadedModel, UploadedNvfp4};

use super::snapshot::PendingTensorUpdateDiagnostics;
use super::stats::tensor_update_stats;
use crate::training::diagnostics::TensorUpdateDiagnostics;

pub(in crate::training::diagnostics) fn finish_update_snapshots(
    stream: &CudaStream,
    uploaded: &UploadedModel,
    pending: Vec<PendingTensorUpdateDiagnostics>,
) -> AppResult<Vec<TensorUpdateDiagnostics>> {
    let mut updates = Vec::with_capacity(pending.len());
    let mut iter = pending.into_iter();

    finish_update(
        &mut updates,
        stream,
        &uploaded.token_embedding,
        iter.next().unwrap(),
    )?;
    finish_layer_norm(&mut updates, stream, &uploaded.ln_f, &mut iter)?;

    for block in &uploaded.blocks {
        finish_layer_norm(&mut updates, stream, &block.ln_1, &mut iter)?;
        finish_linear(&mut updates, stream, &block.attn_qkv, &mut iter)?;
        finish_linear(&mut updates, stream, &block.attn_c_proj, &mut iter)?;
        finish_layer_norm(&mut updates, stream, &block.ln_2, &mut iter)?;
        finish_linear(&mut updates, stream, &block.mlp_up, &mut iter)?;
        finish_linear(&mut updates, stream, &block.mlp_down, &mut iter)?;
    }

    Ok(updates)
}

fn finish_layer_norm(
    updates: &mut Vec<TensorUpdateDiagnostics>,
    stream: &CudaStream,
    layer_norm: &UploadedLayerNorm,
    iter: &mut impl Iterator<Item = PendingTensorUpdateDiagnostics>,
) -> AppResult {
    finish_update(updates, stream, &layer_norm.weight, iter.next().unwrap())?;
    finish_update(updates, stream, &layer_norm.bias, iter.next().unwrap())
}

fn finish_linear(
    updates: &mut Vec<TensorUpdateDiagnostics>,
    stream: &CudaStream,
    linear: &UploadedLinear,
    iter: &mut impl Iterator<Item = PendingTensorUpdateDiagnostics>,
) -> AppResult {
    finish_update(updates, stream, &linear.weight, iter.next().unwrap())?;
    finish_update(updates, stream, &linear.bias, iter.next().unwrap())
}

fn finish_update(
    updates: &mut Vec<TensorUpdateDiagnostics>,
    stream: &CudaStream,
    tensor: &UploadedNvfp4,
    pending: PendingTensorUpdateDiagnostics,
) -> AppResult {
    let after_bytes = tensor.bytes.to_host_vec(stream)?;
    let after_scales = tensor.scales.to_host_vec(stream)?;
    let after_global = tensor.global_scale_to_host(stream)?;
    updates.push(tensor_update_stats(
        pending,
        after_bytes,
        after_scales,
        after_global,
    ));
    Ok(())
}
