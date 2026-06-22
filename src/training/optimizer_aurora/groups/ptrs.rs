use cuda_core::DeviceBuffer;

use crate::upload::{UploadedModel, UploadedNvfp4};

use super::super::super::grads::BackwardBuffers;
use super::super::super::next_latent::NextLatGradBuffers;
use super::super::super::optimizer_state::{AuroraState, OptimizerStateBuffers};
use super::HostPtrs;

pub(super) fn qkv(
    uploaded: &UploadedModel,
    grads: &BackwardBuffers,
    state: &OptimizerStateBuffers,
    i: usize,
) -> HostPtrs {
    linear(
        &uploaded.blocks[i].attn_qkv.weight,
        &grads.blocks[i].d_attn_qkv_weight,
        &state.blocks[i].attn_qkv.weight_aurora,
    )
}

pub(super) fn c_proj(
    uploaded: &UploadedModel,
    grads: &BackwardBuffers,
    state: &OptimizerStateBuffers,
    i: usize,
) -> HostPtrs {
    linear(
        &uploaded.blocks[i].attn_c_proj.weight,
        &grads.blocks[i].d_attn_c_proj_weight,
        &state.blocks[i].attn_c_proj.weight_aurora,
    )
}

pub(super) fn mlp_up(
    uploaded: &UploadedModel,
    grads: &BackwardBuffers,
    state: &OptimizerStateBuffers,
    i: usize,
) -> HostPtrs {
    linear(
        &uploaded.blocks[i].mlp_up.weight,
        &grads.blocks[i].d_mlp_c_fc_weight,
        &state.blocks[i].mlp_up.weight_aurora,
    )
}

pub(super) fn mlp_down(
    uploaded: &UploadedModel,
    grads: &BackwardBuffers,
    state: &OptimizerStateBuffers,
    i: usize,
) -> HostPtrs {
    linear(
        &uploaded.blocks[i].mlp_down.weight,
        &grads.blocks[i].d_mlp_c_proj_weight,
        &state.blocks[i].mlp_down.weight_aurora,
    )
}

pub(super) fn next_latent_input_projection(
    uploaded: &UploadedModel,
    grads: &NextLatGradBuffers,
    state: &OptimizerStateBuffers,
) -> HostPtrs {
    linear(
        &uploaded.next_latent.input_projection.weight,
        &grads.d_input_projection_weight,
        &state.next_latent.input_projection.weight_aurora,
    )
}

pub(super) fn next_latent_transition(
    uploaded: &UploadedModel,
    grads: &NextLatGradBuffers,
    state: &OptimizerStateBuffers,
) -> HostPtrs {
    linear(
        &uploaded.next_latent.transition.weight,
        &grads.d_transition_weight,
        &state.next_latent.transition.weight_aurora,
    )
}

pub(super) fn next_latent_output_projection(
    uploaded: &UploadedModel,
    grads: &NextLatGradBuffers,
    state: &OptimizerStateBuffers,
) -> HostPtrs {
    linear(
        &uploaded.next_latent.output_projection.weight,
        &grads.d_output_projection_weight,
        &state.next_latent.output_projection.weight_aurora,
    )
}

fn linear(weight: &UploadedNvfp4, grad: &DeviceBuffer<f32>, state: &AuroraState) -> HostPtrs {
    HostPtrs {
        grad: grad.cu_deviceptr(),
        momentum: state.momentum.cu_deviceptr(),
        z_master: state.z_master.cu_deviceptr(),
        x_master: state.x_master.cu_deviceptr(),
        bytes: weight.bytes.cu_deviceptr(),
        scales: weight.scales.cu_deviceptr(),
        global_scale: weight.global_scale.cu_deviceptr(),
        rows: 0,
        cols: 0,
        learning_rate_multiplier: 1.0,
    }
}
