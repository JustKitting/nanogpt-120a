use cuda_core::{CudaStream, DeviceBuffer};

use crate::AppResult;
use crate::upload::{UploadedBlock, UploadedLinear, UploadedModel};

use super::record::UpdateSnapshotCollector;
use super::snapshot::PendingTensorUpdateDiagnostics;
use crate::training::grad_block::BlockGradBuffers;
use crate::training::grads::BackwardBuffers;
use crate::training::optimizer_state::{BlockState, LinearState, OptimizerStateBuffers};

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
    collect_linear_updates(
        collector,
        &format!("block{index}.attn_qkv"),
        &block.attn_qkv,
        &grad.d_attn_qkv_weight,
        &grad.d_attn_qkv_bias,
        &state.attn_qkv,
    )?;
    collect_linear_updates(
        collector,
        &format!("block{index}.attn_c_proj"),
        &block.attn_c_proj,
        &grad.d_attn_c_proj_weight,
        &grad.d_attn_c_proj_bias,
        &state.attn_c_proj,
    )?;
    collector.push_layer_norm(
        &format!("block{index}.ln_2"),
        &block.ln_2,
        &grad.ln_2,
        &state.ln_2,
    )?;
    collect_linear_updates(
        collector,
        &format!("block{index}.mlp_up"),
        &block.mlp_up,
        &grad.d_mlp_c_fc_weight,
        &grad.d_mlp_c_fc_bias,
        &state.mlp_up,
    )?;
    collect_linear_updates(
        collector,
        &format!("block{index}.mlp_down"),
        &block.mlp_down,
        &grad.d_mlp_c_proj_weight,
        &grad.d_mlp_c_proj_bias,
        &state.mlp_down,
    )
}

fn collect_linear_updates(
    collector: &mut UpdateSnapshotCollector<'_>,
    name: &str,
    linear: &UploadedLinear,
    weight_grad: &DeviceBuffer<f32>,
    bias_grad: &DeviceBuffer<f32>,
    state: &LinearState,
) -> AppResult {
    collector.push_observed(&format!("{name}.weight"), &linear.weight, weight_grad)?;
    collector.push_adam(
        &format!("{name}.bias"),
        &linear.bias,
        bias_grad,
        &state.bias,
    )
}
