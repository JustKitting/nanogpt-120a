use cuda_core::CudaStream;

use crate::upload::{UploadedModel, UploadedNvfp4, UploadedPair};
use crate::AppResult;

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
    finish_pair(&mut updates, stream, &uploaded.ln_f, &mut iter)?;

    for block in &uploaded.blocks {
        finish_pair(&mut updates, stream, &block.ln_1, &mut iter)?;
        finish_pair(&mut updates, stream, &block.attn_qkv, &mut iter)?;
        finish_pair(&mut updates, stream, &block.attn_c_proj, &mut iter)?;
        finish_pair(&mut updates, stream, &block.ln_2, &mut iter)?;
        finish_pair(&mut updates, stream, &block.mlp_up, &mut iter)?;
        finish_pair(&mut updates, stream, &block.mlp_down, &mut iter)?;
    }

    Ok(updates)
}

fn finish_pair(
    updates: &mut Vec<TensorUpdateDiagnostics>,
    stream: &CudaStream,
    pair: &UploadedPair,
    iter: &mut impl Iterator<Item = PendingTensorUpdateDiagnostics>,
) -> AppResult {
    finish_update(updates, stream, &pair.weight, iter.next().unwrap())?;
    finish_update(updates, stream, &pair.bias, iter.next().unwrap())
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
