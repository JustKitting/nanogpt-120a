use cuda_core::DeviceBuffer;

use crate::upload::{UploadedModel, UploadedNvfp4};

use super::super::super::grads::BackwardBuffers;
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
    }
}
