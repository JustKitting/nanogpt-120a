use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::optimizer::{Nvfp4WeightUpdateArgs, OptimizerModule};

use crate::upload::{UploadedModel, UploadedNvfp4};

use super::grad_block::LayerNormGradBuffers;
use super::grads::BackwardBuffers;
use super::optimizer::OptimizerScratch;

pub fn apply_weight_updates(
    stream: &CudaStream,
    optimizer: &OptimizerModule,
    uploaded: &mut UploadedModel,
    grads: &BackwardBuffers,
    scratch: &mut OptimizerScratch,
) -> Result<(), DriverError> {
    update_tensor(
        stream,
        optimizer,
        &mut uploaded.token_embedding,
        &grads.d_lm_head_weight,
        scratch,
    )?;
    update_layer_norm(
        stream,
        optimizer,
        &mut uploaded.ln_f,
        &grads.final_norm,
        scratch,
    )?;

    for (block, grad) in uploaded.blocks.iter_mut().zip(grads.blocks.iter()) {
        update_layer_norm(stream, optimizer, &mut block.ln_1, &grad.ln_1, scratch)?;
        update_tensor(
            stream,
            optimizer,
            &mut block.attn_qkv.weight,
            &grad.d_attn_qkv_weight,
            scratch,
        )?;
        update_tensor(
            stream,
            optimizer,
            &mut block.attn_qkv.bias,
            &grad.d_attn_qkv_bias,
            scratch,
        )?;
        update_tensor(
            stream,
            optimizer,
            &mut block.attn_c_proj.weight,
            &grad.d_attn_c_proj_weight,
            scratch,
        )?;
        update_tensor(
            stream,
            optimizer,
            &mut block.attn_c_proj.bias,
            &grad.d_attn_c_proj_bias,
            scratch,
        )?;
        update_layer_norm(stream, optimizer, &mut block.ln_2, &grad.ln_2, scratch)?;
        update_tensor(
            stream,
            optimizer,
            &mut block.mlp_up.weight,
            &grad.d_mlp_c_fc_weight,
            scratch,
        )?;
        update_tensor(
            stream,
            optimizer,
            &mut block.mlp_up.bias,
            &grad.d_mlp_c_fc_bias,
            scratch,
        )?;
        update_tensor(
            stream,
            optimizer,
            &mut block.mlp_down.weight,
            &grad.d_mlp_c_proj_weight,
            scratch,
        )?;
        update_tensor(
            stream,
            optimizer,
            &mut block.mlp_down.bias,
            &grad.d_mlp_c_proj_bias,
            scratch,
        )?;
    }

    Ok(())
}

fn update_layer_norm(
    stream: &CudaStream,
    optimizer: &OptimizerModule,
    layer_norm: &mut crate::upload::UploadedLayerNorm,
    grads: &LayerNormGradBuffers,
    scratch: &mut OptimizerScratch,
) -> Result<(), DriverError> {
    update_tensor(
        stream,
        optimizer,
        &mut layer_norm.weight,
        &grads.d_weight,
        scratch,
    )?;
    update_tensor(
        stream,
        optimizer,
        &mut layer_norm.bias,
        &grads.d_bias,
        scratch,
    )
}

fn update_tensor(
    stream: &CudaStream,
    optimizer: &OptimizerModule,
    tensor: &mut UploadedNvfp4,
    update: &DeviceBuffer<f32>,
    scratch: &mut OptimizerScratch,
) -> Result<(), DriverError> {
    let len = (tensor.bytes.len() * 2) as u32;
    optimizer.apply_nvfp4_weight_update(Nvfp4WeightUpdateArgs {
        stream,
        bytes: &mut tensor.bytes,
        scales: &mut tensor.scales,
        global_scale: tensor.global_scale,
        aurora_update: update,
        fp32_workspace: &mut scratch.fp32_workspace,
        amax: &mut scratch.amax,
        next_global_scale: &mut scratch.next_global_scale,
        len,
        learning_rate: 1.0e-4,
        weight_decay: 0.0,
    })?;

    tensor.global_scale = scratch.next_global_scale.to_host_vec(stream)?[0];
    Ok(())
}
