use cuda_core::{CudaStream, DriverError};
use gpt2_nvfp4::GPT2_N_LAYER;
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;

use super::tensor::{AdamState, AuroraState, StateInit};
use crate::{
    training::{
        device_buffer::block_array,
        learning_rate::schedule_free_average_coefficient,
        update_skip::{UpdateSkipDecision, UpdateSkipState},
    },
    upload::{UploadedBlock, UploadedLayerNorm, UploadedLinear, UploadedModel, UploadedNextLat},
};

pub struct OptimizerStateBuffers {
    pub(in crate::training) step: u32,
    pub(in crate::training) schedule_free_weight_sum: f32,
    pub(in crate::training) update_skip: UpdateSkipState,
    pub(in crate::training) token_embedding: AdamState,
    pub(in crate::training) ln_f: LayerNormState,
    pub(in crate::training) next_latent: NextLatState,
    pub(in crate::training) blocks: [BlockState; GPT2_N_LAYER],
}

impl OptimizerStateBuffers {
    pub fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        uploaded: &UploadedModel,
    ) -> Result<Self, DriverError> {
        let init = StateInit::new(stream, decode);
        Ok(Self {
            step: 0,
            schedule_free_weight_sum: 0.0,
            update_skip: UpdateSkipState::new(),
            token_embedding: AdamState::new(init, &uploaded.token_embedding)?,
            ln_f: LayerNormState::new(init, &uploaded.ln_f)?,
            next_latent: NextLatState::new(init, &uploaded.next_latent)?,
            blocks: block_array(|i| BlockState::new(init, &uploaded.blocks[i]))?,
        })
    }

    pub(in crate::training) fn advance(&mut self) -> u32 {
        self.step += 1;
        self.step
    }

    pub(in crate::training) fn schedule_free_average_coefficient(&mut self, step: u32) -> f32 {
        schedule_free_average_coefficient(step, &mut self.schedule_free_weight_sum)
    }

    pub(in crate::training) fn next_step(&self) -> u32 {
        self.step + 1
    }

    pub(in crate::training) fn should_skip_update(
        &mut self,
        loss: Option<f32>,
        grad_norm: f32,
    ) -> UpdateSkipDecision {
        self.update_skip.observe(loss, grad_norm)
    }
}

pub(in crate::training) struct BlockState {
    pub(in crate::training) ln_1: LayerNormState,
    pub(in crate::training) attn_qkv: LinearState,
    pub(in crate::training) attn_c_proj: LinearState,
    pub(in crate::training) ln_2: LayerNormState,
    pub(in crate::training) mlp_up: LinearState,
    pub(in crate::training) mlp_down: LinearState,
}

impl BlockState {
    fn new(init: StateInit<'_>, block: &UploadedBlock) -> Result<Self, DriverError> {
        Ok(Self {
            ln_1: LayerNormState::new(init, &block.ln_1)?,
            attn_qkv: LinearState::new(init, &block.attn_qkv)?,
            attn_c_proj: LinearState::new(init, &block.attn_c_proj)?,
            ln_2: LayerNormState::new(init, &block.ln_2)?,
            mlp_up: LinearState::new(init, &block.mlp_up)?,
            mlp_down: LinearState::new(init, &block.mlp_down)?,
        })
    }
}

pub(in crate::training) struct NextLatState {
    pub(in crate::training) norm: LayerNormState,
    pub(in crate::training) input_projection: LinearState,
    pub(in crate::training) transition: LinearState,
    pub(in crate::training) output_projection: LinearState,
}

impl NextLatState {
    fn new(init: StateInit<'_>, next_latent: &UploadedNextLat) -> Result<Self, DriverError> {
        Ok(Self {
            norm: LayerNormState::new(init, &next_latent.norm)?,
            input_projection: LinearState::new(init, &next_latent.input_projection)?,
            transition: LinearState::new(init, &next_latent.transition)?,
            output_projection: LinearState::new(init, &next_latent.output_projection)?,
        })
    }
}

pub(in crate::training) struct LayerNormState {
    pub(in crate::training) weight: AdamState,
    pub(in crate::training) bias: AdamState,
}

impl LayerNormState {
    fn new(init: StateInit<'_>, layer_norm: &UploadedLayerNorm) -> Result<Self, DriverError> {
        Ok(Self {
            weight: AdamState::new(init, &layer_norm.weight)?,
            bias: AdamState::new(init, &layer_norm.bias)?,
        })
    }
}

pub(in crate::training) struct LinearState {
    pub(in crate::training) weight_aurora: AuroraState,
    pub(in crate::training) bias: AdamState,
}

impl LinearState {
    fn new(init: StateInit<'_>, linear: &UploadedLinear) -> Result<Self, DriverError> {
        Ok(Self {
            weight_aurora: AuroraState::new(init, &linear.weight)?,
            bias: AdamState::new(init, &linear.bias)?,
        })
    }
}
