use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;

use super::{
    AdamState, AuroraState, BlockState, LayerNormState, LinearState, NextLatState,
    OptimizerStateBuffers,
    device::{block_array, clone_device, decode_master},
};
use crate::{
    training::update_skip::UpdateSkipState,
    upload::{
        UploadedBlock, UploadedLayerNorm, UploadedLinear, UploadedModel, UploadedNextLat,
        UploadedNvfp4,
    },
};

impl OptimizerStateBuffers {
    pub fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        uploaded: &UploadedModel,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            step: 0,
            schedule_free_weight_sum: 0.0,
            update_skip: UpdateSkipState::new(),
            token_embedding: AdamState::new(stream, decode, &uploaded.token_embedding)?,
            ln_f: LayerNormState::new(stream, decode, &uploaded.ln_f)?,
            next_latent: NextLatState::new(stream, decode, &uploaded.next_latent)?,
            blocks: block_array(|i| BlockState::new(stream, decode, &uploaded.blocks[i]))?,
        })
    }
}

impl BlockState {
    fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        block: &UploadedBlock,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            ln_1: LayerNormState::new(stream, decode, &block.ln_1)?,
            attn_qkv: LinearState::new(stream, decode, &block.attn_qkv)?,
            attn_c_proj: LinearState::new(stream, decode, &block.attn_c_proj)?,
            ln_2: LayerNormState::new(stream, decode, &block.ln_2)?,
            mlp_up: LinearState::new(stream, decode, &block.mlp_up)?,
            mlp_down: LinearState::new(stream, decode, &block.mlp_down)?,
        })
    }
}

impl NextLatState {
    fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        next_latent: &UploadedNextLat,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            norm: LayerNormState::new(stream, decode, &next_latent.norm)?,
            input_projection: LinearState::new(stream, decode, &next_latent.input_projection)?,
            transition: LinearState::new(stream, decode, &next_latent.transition)?,
            output_projection: LinearState::new(stream, decode, &next_latent.output_projection)?,
        })
    }
}

impl LayerNormState {
    fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        layer_norm: &UploadedLayerNorm,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            weight: AdamState::new(stream, decode, &layer_norm.weight)?,
            bias: AdamState::new(stream, decode, &layer_norm.bias)?,
        })
    }
}

impl LinearState {
    fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        linear: &UploadedLinear,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            weight_aurora: AuroraState::new(stream, decode, &linear.weight)?,
            bias: AdamState::new(stream, decode, &linear.bias)?,
        })
    }
}

impl AdamState {
    fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        tensor: &UploadedNvfp4,
    ) -> Result<Self, DriverError> {
        let master = decode_master(stream, decode, tensor)?;
        Ok(Self {
            z_master: clone_device(stream, &master)?,
            x_master: master,
            first: DeviceBuffer::zeroed(stream, tensor.len)?,
            second: DeviceBuffer::zeroed(stream, tensor.len)?,
        })
    }
}

impl AuroraState {
    fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        tensor: &UploadedNvfp4,
    ) -> Result<Self, DriverError> {
        let master = decode_master(stream, decode, tensor)?;
        Ok(Self {
            z_master: clone_device(stream, &master)?,
            x_master: master,
            momentum: DeviceBuffer::zeroed(stream, tensor.len)?,
        })
    }
}
