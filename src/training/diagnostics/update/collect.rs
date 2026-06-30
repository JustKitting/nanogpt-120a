use cuda_core::CudaStream;

use crate::AppResult;
use crate::upload::{UploadedBlock, UploadedModel};

use super::record::UpdateSnapshotCollector;
use super::snapshot::PendingTensorUpdateDiagnostics;
use crate::training::grad_block::BlockGradBuffers;
use crate::training::grads::BackwardBuffers;
use crate::training::optimizer_state::{BlockState, OptimizerStateBuffers};

pub(in crate::training::diagnostics) fn collect_update_snapshots(
    stream: &CudaStream,
    uploaded: &UploadedModel,
    grads: &BackwardBuffers,
    state: &OptimizerStateBuffers,
    step: u32,
    average_coefficient: f32,
) -> AppResult<Vec<PendingTensorUpdateDiagnostics>> {
    let mut collector = UpdateSnapshotCollector::new(stream, step, average_coefficient);
    collector.push_adam(
        "token_embedding",
        &uploaded.token_embedding,
        &grads.d_lm_head_weight,
        &state.token_embedding,
    )?;
    collector.push_layer_norm("ln_f", &uploaded.ln_f, &grads.final_norm, &state.ln_f)?;

    for (index, ((block, grad), state)) in uploaded
        .blocks
        .iter()
        .zip(grads.blocks.iter())
        .zip(state.blocks.iter())
        .enumerate()
    {
        collect_block_updates(&mut collector, index, block, grad, state)?;
    }

    Ok(collector.into_updates())
}

fn collect_block_updates(
    collector: &mut UpdateSnapshotCollector<'_>,
    index: usize,
    block: &UploadedBlock,
    grad: &BlockGradBuffers,
    state: &BlockState,
) -> AppResult {
    collector.push_layer_norm(
        &format!("block{index}.ln_1"),
        &block.ln_1,
        &grad.ln_1,
        &state.ln_1,
    )?;
    collector.push_observed(
        &format!("block{index}.attn_qkv.weight"),
        &block.attn_qkv.weight,
        &grad.d_attn_qkv_weight,
    )?;
    collector.push_adam(
        &format!("block{index}.attn_qkv.bias"),
        &block.attn_qkv.bias,
        &grad.d_attn_qkv_bias,
        &state.attn_qkv.bias,
    )?;
    collector.push_observed(
        &format!("block{index}.attn_c_proj.weight"),
        &block.attn_c_proj.weight,
        &grad.d_attn_c_proj_weight,
    )?;
    collector.push_adam(
        &format!("block{index}.attn_c_proj.bias"),
        &block.attn_c_proj.bias,
        &grad.d_attn_c_proj_bias,
        &state.attn_c_proj.bias,
    )?;
    collector.push_layer_norm(
        &format!("block{index}.ln_2"),
        &block.ln_2,
        &grad.ln_2,
        &state.ln_2,
    )?;
    collector.push_observed(
        &format!("block{index}.mlp_up.weight"),
        &block.mlp_up.weight,
        &grad.d_mlp_c_fc_weight,
    )?;
    collector.push_adam(
        &format!("block{index}.mlp_up.bias"),
        &block.mlp_up.bias,
        &grad.d_mlp_c_fc_bias,
        &state.mlp_up.bias,
    )?;
    collector.push_observed(
        &format!("block{index}.mlp_down.weight"),
        &block.mlp_down.weight,
        &grad.d_mlp_c_proj_weight,
    )?;
    collector.push_adam(
        &format!("block{index}.mlp_down.bias"),
        &block.mlp_down.bias,
        &grad.d_mlp_c_proj_bias,
        &state.mlp_down.bias,
    )
}
